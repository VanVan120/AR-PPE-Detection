"""Honest evaluation of the assembly-order mistake detector.

There is no fine-grained mistake ground truth on disk (that is a separate gated
download), so we cannot report the *canonical* Assembly101 mistake benchmark. Instead
we evaluate exactly what this detector claims to do — catch **order violations of the
learned prerequisites** — with a self-supervised protocol on the held-out val fold:

  * **Clean pass (false positives).** Every val sequence is assumed correct. Run it
    through the monitor; a flag here is a false positive. Report the sequence-level
    false-positive rate (FPR) and the per-transition FP rate. Low is good.
  * **Injected-fault pass (recall).** For each val sequence, take a learned constraint
    `a -> b` (a must precede b) whose two steps both appear in the sequence in the
    correct order, and **swap them** so b now precedes a. That is a guaranteed
    prerequisite violation. Fraction detected = recall.
  * **Precision** over the mixed set: perturbed sequences are positives, clean
    sequences negatives; precision = detected_perturbed / (detected_perturbed +
    flagged_clean).

The model is learned on **train** and evaluated on **val** (no leakage). This measures
the detector's own claim, and is explicitly NOT the fine-grained benchmark — stated so
no one mistakes it for one.

    python -m phase3_activity.tas.mistake_eval --data-root phase3_activity/data
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .procedure import (MistakeMonitor, ProcedureModel, load_step_sequences,
                        score_sequence)


def _positions(seq: Sequence[int]) -> Dict[int, List[int]]:
    pos: Dict[int, List[int]] = {}
    for i, s in enumerate(seq):
        pos.setdefault(s, []).append(i)
    return pos


def perturbations_for(model: ProcedureModel, seq: Sequence[int],
                      max_per_seq: int, rng: random.Random) -> List[Tuple[List[int], Tuple[int, int]]]:
    """Build up to `max_per_seq` guaranteed order violations by swapping a constraint
    pair (a before b, a -> b required) that both occur exactly once in `seq`. Returns
    [(perturbed_sequence, (a, b)), ...]."""
    pos = _positions(seq)
    candidates: List[Tuple[int, int]] = []
    for (a, b) in model.constraint_stats:                       # a must precede b
        if a in pos and b in pos and len(pos[a]) == 1 and len(pos[b]) == 1:
            if pos[a][0] < pos[b][0]:                           # currently correct order
                candidates.append((a, b))
    rng.shuffle(candidates)
    out: List[Tuple[List[int], Tuple[int, int]]] = []
    for a, b in candidates[:max_per_seq]:
        pa, pb = pos[a][0], pos[b][0]
        pert = list(seq)
        pert[pa], pert[pb] = pert[pb], pert[pa]                 # swap -> b now before a
        out.append((pert, (a, b)))
    return out


def _flagged(model: ProcedureModel, seq: Sequence[int], use_novel: bool,
             kinds: Optional[set] = None) -> bool:
    """True if the monitor raises any event (optionally restricted to `kinds`)."""
    events = score_sequence(model, seq, use_novel_transition=use_novel)
    if kinds is None:
        return len(events) > 0
    return any(ev.kind in kinds for _i, ev in events)


def evaluate(model: ProcedureModel, val_seqs: Sequence[Tuple[str, List[int]]],
             max_per_seq: int = 2, use_novel: bool = False, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n_clean = len(val_seqs)
    clean_flagged = 0
    clean_events = 0
    clean_transitions = 0
    n_pert = 0
    pert_detected = 0
    order_kinds = {"order_violation"}                           # the injected fault type

    for _name, seq in val_seqs:
        # clean
        events = score_sequence(model, seq, use_novel_transition=use_novel)
        if events:
            clean_flagged += 1
        clean_events += len(events)
        clean_transitions += len(list(dict.fromkeys(seq)))      # ~#transitions (distinct steps)
        # injected faults
        for pert, _pair in perturbations_for(model, seq, max_per_seq, rng):
            n_pert += 1
            if _flagged(model, pert, use_novel, kinds=order_kinds):
                pert_detected += 1

    fpr_seq = clean_flagged / max(n_clean, 1)
    fpr_trans = clean_events / max(clean_transitions, 1)
    recall = pert_detected / max(n_pert, 1)
    precision = pert_detected / max(pert_detected + clean_flagged, 1)
    return {
        "n_clean": n_clean, "clean_flagged": clean_flagged,
        "fpr_seq": fpr_seq, "fpr_per_transition": fpr_trans,
        "n_perturbations": n_pert, "detected": pert_detected,
        "recall": recall, "precision": precision, "use_novel_transition": use_novel,
    }


def main(argv=None) -> int:
    import argparse

    from .dataset import load_actions_csv

    ap = argparse.ArgumentParser(description="Evaluate the assembly-order mistake detector")
    ap.add_argument("--data-root", default="phase3_activity/data")
    ap.add_argument("--model", default="phase3_activity/models/procedure_model.json",
                    help="a saved procedure model; if absent, one is fit from the train fold")
    ap.add_argument("--procedure", default="assembly",
                    choices=["assembly", "disassembly", "both"],
                    help="which workflow to evaluate (must match how --model was built)")
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--precedence-tau", type=float, default=0.99)
    ap.add_argument("--max-per-seq", type=int, default=2,
                    help="max injected order violations per val sequence")
    ap.add_argument("--novel", action="store_true",
                    help="also count the weaker novel-transition signal (raises recall AND FPR)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ann = os.path.join(args.data_root, "coarse-annotations")
    actions_dict, id_to_action = load_actions_csv(os.path.join(ann, "actions.csv"))
    splits, labels = os.path.join(ann, "coarse_splits"), os.path.join(ann, "coarse_labels")

    if os.path.isfile(args.model):
        model = ProcedureModel.load(args.model)
        src = f"loaded {args.model}"
    else:
        train_seqs = load_step_sequences(splits, labels, "train", actions_dict,
                                         procedure=args.procedure)
        model = ProcedureModel.fit([s for _n, s in train_seqs], min_support=args.min_support,
                                   precedence_tau=args.precedence_tau, id_to_action=id_to_action)
        src = f"fit from train ({model.n_videos} videos, {args.procedure})"

    val_seqs = load_step_sequences(splits, labels, "val", actions_dict, procedure=args.procedure)
    if not val_seqs:
        print("no val sequences found — check --data-root"); return 1

    r = evaluate(model, val_seqs, max_per_seq=args.max_per_seq, use_novel=args.novel,
                 seed=args.seed)
    print(f"procedure model: {src}; {len(model.constraint_stats)} constraints")
    print(f"val sequences: {r['n_clean']}")
    print("--- clean pass (false positives; lower is better) ---")
    print(f"  sequences flagged: {r['clean_flagged']}/{r['n_clean']}  "
          f"(FPR {r['fpr_seq'] * 100:.1f}%)")
    print(f"  per-transition FP rate: {r['fpr_per_transition'] * 100:.2f}%")
    print("--- injected order-violation pass (recall; higher is better) ---")
    print(f"  detected: {r['detected']}/{r['n_perturbations']}  (recall {r['recall'] * 100:.1f}%)")
    print("--- combined ---")
    print(f"  precision {r['precision'] * 100:.1f}%   recall {r['recall'] * 100:.1f}%   "
          f"(novel_transition={'on' if r['use_novel_transition'] else 'off'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

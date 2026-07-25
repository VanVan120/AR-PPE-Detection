"""Phase 3 end-to-end workflow monitor -- watch all three capabilities run together.

The other Phase 3 CLIs each do ONE thing: `train`/`evaluate` score step recognition,
`procedure` builds the order model, `anticipation` builds + scores the next-step model,
`mistake_eval` runs the injected-fault protocol. None of them shows the actual product
-- *dynamic workflow monitoring* -- which is all three running on one step stream:

    recognised step  ->  is it out of order?  ->  what comes next?

That is what this module does. It replays a step stream one step at a time through the
`MistakeMonitor` and the `AnticipationModel` exactly as the live Phase 2 seam would,
and prints a monitor trace. This is the "offline replay of a recognised/GT step stream"
that the README calls the meaningful path today (live labels still depend on a real TSM
extractor -- see `infer_seam.py`).

Three stream sources, in descending order of fidelity:

  * `--source model` -- the TRAINED TAS checkpoint predicts the steps from the real
    Assembly101 features, then those predictions drive the monitor. This is the true
    end-to-end path (recognition -> mistake -> anticipation) and needs the LMDB
    features + a checkpoint.
  * `--source gt` -- the ground-truth step stream from the coarse annotations (2.6 MB,
    no features needed). Isolates the workflow logic from recogniser error.
  * `--source sample` -- no data at all: walk the learned model's own most-likely chain.
    Demonstrates the MECHANISM only; no accuracy is reported (it would be circular).

`--inject-fault` swaps a learned constraint pair (a must precede b -> b first), a
guaranteed order violation, so you can watch the detector catch a known-bad build.

    python -m phase3_activity.tas.demo                        # auto-picks the best source
    python -m phase3_activity.tas.demo --inject-fault
    python -m phase3_activity.tas.demo --source model --index 0
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .anticipation import AnticipationModel
from .procedure import (MistakeEvent, MistakeMonitor, ProcedureModel,
                        load_step_sequences, reduce_segments_to_step_ids)

TOPK = 3
RULE = "-" * 78


# --- stream sources -----------------------------------------------------------
def stream_from_gt(data_root: str, fold: str, procedure: str, index: int,
                   name: str = "") -> Tuple[str, List[int]]:
    """Ground-truth step stream for one annotation (coarse annotations only)."""
    from .dataset import load_actions_csv
    ann = os.path.join(data_root, "coarse-annotations")
    actions_dict, _ = load_actions_csv(os.path.join(ann, "actions.csv"))
    seqs = load_step_sequences(os.path.join(ann, "coarse_splits"),
                               os.path.join(ann, "coarse_labels"), fold, actions_dict,
                               procedure=procedure)
    if not seqs:
        raise SystemExit(f"no {procedure} sequences in fold '{fold}' under {ann}")
    if name:
        match = [s for s in seqs if s[0] == name or s[0] == f"{name}.txt"]
        if not match:
            raise SystemExit(f"sequence '{name}' not found in fold '{fold}'")
        return match[0]
    return seqs[index % len(seqs)]


def stream_from_model(data_root: str, features_root: str, view: str, fold: str,
                      procedure: str, ckpt: str, index: int,
                      name: str = "") -> Tuple[str, List[int]]:
    """Run the TRAINED TAS model over one sequence's real features and collapse its
    per-frame predictions into a step stream. The true end-to-end path."""
    import numpy as np
    import torch

    from .dataset import (LmdbFeatureStore, build_samples, keep_present,
                          load_actions_csv, video_ids_for_fold)
    from .model import load_model
    from .postprocess import upsample_predictions
    from .torch_dataset import TASDataset

    ann = os.path.join(data_root, "coarse-annotations")
    actions_dict, id_to_action = load_actions_csv(os.path.join(ann, "actions.csv"))
    store = LmdbFeatureStore(features_root or data_root)
    names = video_ids_for_fold(os.path.join(ann, "coarse_splits"), fold)
    if procedure in ("assembly", "disassembly"):
        names = [n for n in names if n.startswith(f"{procedure}_")]
    samples = keep_present(store, view,
                           build_samples(os.path.join(ann, "coarse_labels"), names, actions_dict))
    if not samples:
        raise SystemExit(f"no {procedure} features present for view '{view}' -- "
                         f"is the LMDB under {features_root or data_root}?")
    if name:
        pick = [i for i, s in enumerate(samples)
                if name in (s.get("name"), s.get("core"), f"{s.get('name')}.txt")]
        if not pick:
            raise SystemExit(f"sequence '{name}' has no features in view '{view}'")
        index = pick[0]
    index %= len(samples)

    ds = TASDataset(samples, view, store, 20, 1200)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)
    model.eval()
    x, _y, vid, orig_len = ds[index]
    with torch.no_grad():
        outputs = model(x.unsqueeze(0).to(device))
    pooled = outputs[-1, 0].argmax(0).cpu().numpy()
    frames = upsample_predictions(pooled, orig_len, getattr(ds, "chunk_size", 20))
    # per-frame class ids -> ordered step stream (collapse consecutive duplicates),
    # mirroring how reduce_segments_to_step_ids treats a segment list.
    steps: List[int] = []
    for c in np.asarray(frames).tolist():
        if not steps or steps[-1] != int(c):
            steps.append(int(c))
    return f"{vid} [predicted by {os.path.basename(ckpt)}]", steps


def stream_sampled(antic: AnticipationModel, length: int = 12) -> Tuple[str, List[int]]:
    """Zero-download fallback: walk the learned model's most-likely chain from its
    most-likely start step. Mechanism demo only -- never scored.

    Sampling uses `use_feasibility=True` (prerequisite masking) even though prediction
    does not. These are different tasks: predicting what a human will do next is hurt by
    strict prereqs (assembly order is loose), but GENERATING a reference plan must respect
    them -- otherwise the greedy walk picks a terminal step like 'demonstrate
    functionality' early and every later step trips an order violation, which would make
    a clean demo look broken. With the mask on, the generated chain is violation-free by
    construction, so `--inject-fault` shows exactly one flag.
    """
    if antic.starts:
        cur = max(antic.starts.items(), key=lambda t: (t[1], -t[0]))[0]
    elif antic.vocab:
        cur = sorted(antic.vocab)[0]
    else:
        raise SystemExit("anticipation model has no vocabulary -- rebuild it")
    steps = [int(cur)]
    for _ in range(length - 1):
        nxt = antic.predict_next(steps, 1, use_feasibility=True)
        if not nxt:
            break
        steps.append(int(nxt[0][0]))
    return "sampled from the learned model (no annotations found)", steps


def inject_fault(model: ProcedureModel, steps: Sequence[int], seed: int = 0
                 ) -> Tuple[List[int], Optional[Tuple[int, int]], int, str]:
    """Plant one guaranteed order violation. Returns (steps, (a, b), index, how).

    Two mechanisms, because one is not enough:

    **Swap** — the evaluation protocol's method (`perturbations_for`): exchange a pair
    a->b that each occur *exactly once*. Unambiguous, which is why the published numbers
    use it, and it leaves the sequence the same length.

    **Insert** — used when no such pair exists. That is not a rare corner: a *recognised*
    step stream repeats steps constantly (the demo's own default sequence has 'attach
    base' three times and 'screw chassis' six), so the exactly-once condition almost never
    holds and the swap silently found nothing. `--inject-fault` then printed a warning and
    replayed an unmodified stream — while the launcher told the user to look for a
    "CAUGHT" line that could never appear. Inserting a fresh `a` just after the first `b`
    plants the violation without needing uniqueness.

    `index` is where the offending step ends up, so the caller can check that the monitor
    flagged *that* step rather than one of the violations the stream already contained.
    """
    from .mistake_eval import perturbations_for
    seq = list(steps)
    rng = random.Random(seed)

    perts = perturbations_for(model, seq, 1, rng)
    if perts:
        pert, (a, b) = perts[0]
        # After the swap `a` sits where `b` was. Both were unique, so this is exact.
        return pert, (a, b), seq.index(b), "swapped"

    # Fall back to insertion: any constraint whose two steps are both present will do.
    present = set(seq)
    pairs = [(a, b) for (a, b) in model.constraint_stats
             if a in present and b in present]
    if not pairs:
        return seq, None, -1, ""
    rng.shuffle(pairs)

    # A recognised stream usually violates order already, so not every insertion point
    # yields *new* evidence: drop a duplicate step next to a position the monitor was
    # going to flag anyway and "CAUGHT" says nothing. So prefer a candidate that both
    # flags at the insertion point AND raises the total count. Scoring is a pure-Python
    # pass over a short list, so trying several costs nothing.
    from .procedure import score_sequence
    baseline = sum(1 for _i, e in score_sequence(model, seq)
                   if e.kind == "order_violation")
    fallback = None
    for a, b in pairs:
        for j, step in enumerate(seq):
            if step != b:
                continue
            at = j + 1
            pert = seq[:at] + [a] + seq[at:]
            events = score_sequence(model, pert)
            here = any(i == at and e.kind == "order_violation" for i, e in events)
            if not here:
                continue
            if fallback is None:
                fallback = (pert, (a, b), at)
            if sum(1 for _i, e in events if e.kind == "order_violation") > baseline:
                return pert, (a, b), at, "inserted"
    if fallback is not None:
        return fallback[0], fallback[1], fallback[2], "inserted"
    a, b = pairs[0]
    at = seq.index(b) + 1
    return seq[:at] + [a] + seq[at:], (a, b), at, "inserted"


# --- the monitor replay -------------------------------------------------------
def _fmt_pred(antic: Optional[AnticipationModel], history: Sequence[int], k: int) -> str:
    if antic is None:
        return ""
    top = antic.predict_next(history, k)
    if not top:
        return "next: (no prediction)"
    return "next: " + ", ".join(f"{antic.name(n)} {p * 100:.0f}%" for n, p in top)


def replay(steps: Sequence[int], proc: ProcedureModel,
           antic: Optional[AnticipationModel], k: int = TOPK,
           score: bool = True, quiet: bool = False) -> dict:
    """Feed `steps` through the monitor + anticipator one at a time, printing the trace.

    Anticipation is scored honestly: at position i the prediction is made from
    `steps[:i]` only (the history the live system would have), then the true step is
    revealed. Position 0 is the cold start.
    """
    monitor = MistakeMonitor(proc)
    events: List[Tuple[int, MistakeEvent]] = []
    hit1 = hit3 = total = 0

    for i, step in enumerate(steps):
        history = list(steps[:i])
        marker = "     "
        if antic is not None and score:
            top = [n for n, _p in antic.predict_next(history, k)]
            total += 1
            if top[:1] == [step]:
                hit1 += 1
                hit3 += 1
                marker = "<-#1 "
            elif step in top[:k]:
                hit3 += 1
                marker = f"<-top{k}"
        ev = monitor.observe(step)
        if ev is not None:
            events.append((i, ev))
        if not quiet:
            print(f"  [{i + 1:02d}] {proc.name(step)[:34]:34s} {marker} "
                  f"{_fmt_pred(antic, list(steps[:i + 1]), k)}")
            if ev is not None:
                print(f"       *** MISTAKE ({ev.kind}): {ev.detail}")

    return {"n_steps": len(steps), "events": events,
            "top1": 100.0 * hit1 / total if total else None,
            "topk": 100.0 * hit3 / total if total else None, "k": k}


# --- gt-stream vs predicted-stream flag rate ----------------------------------
def scan_streams(data_root: str, features_root: str, view: str, fold: str,
                 procedure: str, ckpt: str, proc: ProcedureModel, count: int) -> dict:
    """Replay the SAME sequences twice -- once from ground-truth steps, once from the
    trained recogniser's predicted steps -- and compare how often the order monitor
    fires. Quantifies how much recogniser error inflates mistake flags, which the
    annotation-only `mistake_eval` protocol cannot see (it scores GT streams).
    """
    import numpy as np
    import torch

    from .dataset import (LmdbFeatureStore, build_samples, keep_present,
                          load_actions_csv, video_ids_for_fold)
    from .model import load_model
    from .postprocess import upsample_predictions
    from .procedure import score_sequence
    from .torch_dataset import TASDataset

    ann = os.path.join(data_root, "coarse-annotations")
    actions_dict, _ = load_actions_csv(os.path.join(ann, "actions.csv"))
    gt_seqs = dict(load_step_sequences(os.path.join(ann, "coarse_splits"),
                                       os.path.join(ann, "coarse_labels"), fold,
                                       actions_dict, procedure=procedure))
    store = LmdbFeatureStore(features_root or data_root)
    names = video_ids_for_fold(os.path.join(ann, "coarse_splits"), fold)
    if procedure in ("assembly", "disassembly"):
        names = [n for n in names if n.startswith(f"{procedure}_")]
    # Pair BY NAME, never by index: `keep_present` drops sequences whose features are
    # missing from this view, so the GT list and the feature list are not aligned and
    # index-pairing would silently compare two different recordings.
    samples = [s for s in keep_present(store, view,
                                       build_samples(os.path.join(ann, "coarse_labels"),
                                                     names, actions_dict))
               if s["name"] in gt_seqs][:count]
    if not samples:
        return {}
    ds = TASDataset(samples, view, store, 20, 1200)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)
    model.eval()

    rows = []
    for i, s in enumerate(samples):
        x, _y, vid, orig_len = ds[i]
        with torch.no_grad():
            outputs = model(x.unsqueeze(0).to(device))
        frames = upsample_predictions(outputs[-1, 0].argmax(0).cpu().numpy(), orig_len,
                                      getattr(ds, "chunk_size", 20))
        pr_steps: List[int] = []
        for c in np.asarray(frames).tolist():
            if not pr_steps or pr_steps[-1] != int(c):
                pr_steps.append(int(c))
        gt_steps = gt_seqs[s["name"]]
        rows.append({
            "name": s["name"], "vid": vid,
            "gt_steps": len(gt_steps), "pred_steps": len(pr_steps),
            "gt_flags": len(score_sequence(proc, gt_steps)),
            "pred_flags": len(score_sequence(proc, pr_steps)),
        })
    if not rows:
        return {}
    # Same denominator convention as `mistake_eval`: n entries -> n-1 transitions, so the
    # GT column here reproduces that CLI's per-transition FP rate exactly.
    gt_t = sum(max(r["gt_steps"] - 1, 0) for r in rows)
    pr_t = sum(max(r["pred_steps"] - 1, 0) for r in rows)
    return {
        "n": len(rows),
        "gt_transitions": gt_t, "pred_transitions": pr_t,
        "gt_flags": sum(r["gt_flags"] for r in rows),
        "pred_flags": sum(r["pred_flags"] for r in rows),
        "gt_fpr": 100.0 * sum(r["gt_flags"] for r in rows) / max(gt_t, 1),
        "pred_fpr": 100.0 * sum(r["pred_flags"] for r in rows) / max(pr_t, 1),
        "gt_seq_flagged": 100.0 * sum(1 for r in rows if r["gt_flags"]) / len(rows),
        "pred_seq_flagged": 100.0 * sum(1 for r in rows if r["pred_flags"]) / len(rows),
        "rows": rows,
    }


# --- CLI ----------------------------------------------------------------------
def _load_models(models_dir: str, data_root: str, procedure: str,
                 ) -> Tuple[ProcedureModel, Optional[AnticipationModel]]:
    """Load the built JSON models; if absent, rebuild them from the coarse annotations."""
    ppath = os.path.join(models_dir, "procedure_model.json")
    apath = os.path.join(models_dir, "anticipation_model.json")
    if os.path.isfile(ppath):
        proc = ProcedureModel.load(ppath)
    else:
        ann = os.path.join(data_root, "coarse-annotations")
        if not os.path.isdir(ann):
            raise SystemExit(
                f"no order model at {ppath} and no annotations at {ann}.\n"
                f"Build one with:  python -m phase3_activity.tas.procedure")
        from .dataset import load_actions_csv
        actions_dict, id_to_action = load_actions_csv(os.path.join(ann, "actions.csv"))
        seqs = load_step_sequences(os.path.join(ann, "coarse_splits"),
                                   os.path.join(ann, "coarse_labels"), "train",
                                   actions_dict, procedure=procedure)
        proc = ProcedureModel.fit([s for _n, s in seqs], id_to_action=id_to_action)
        print(f"[info] built the order model on the fly ({len(seqs)} train sequences)")
    antic = AnticipationModel.load(apath) if os.path.isfile(apath) else None
    return proc, antic


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Phase 3 workflow monitor: replay a step stream through step "
                    "recognition + mistake detection + next-step anticipation.")
    ap.add_argument("--data-root", default="phase3_activity/data")
    ap.add_argument("--features-root", default="", help="dir holding per-view LMDBs")
    ap.add_argument("--models-dir", default="phase3_activity/models")
    ap.add_argument("--source", default="auto", choices=["auto", "model", "gt", "sample"],
                    help="where the step stream comes from (default: auto -- the best "
                         "available: model > gt > sample)")
    ap.add_argument("--ckpt", default="phase3_activity/models/mstcn_best.pt")
    ap.add_argument("--view", default="C10095_rgb")
    ap.add_argument("--fold", default="val", choices=["train", "train_val", "val"])
    ap.add_argument("--procedure", default="assembly",
                    choices=["assembly", "disassembly", "both"])
    ap.add_argument("--index", type=int, default=0, help="which sequence in the fold")
    ap.add_argument("--name", default="", help="pick a sequence by name instead of index")
    ap.add_argument("--inject-fault", action="store_true",
                    help="swap a learned constraint pair to force an order violation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--scan", type=int, default=0, metavar="N",
                    help="instead of one replay, compare ground-truth vs "
                         "recogniser-predicted step streams over N sequences and report "
                         "how much recogniser error inflates the mistake-flag rate")
    args = ap.parse_args(argv)

    proc, antic = _load_models(args.models_dir, args.data_root, args.procedure)

    if args.scan:
        s = scan_streams(args.data_root, args.features_root, args.view, args.fold,
                         args.procedure, args.ckpt, proc, args.scan)
        if not s:
            raise SystemExit("scan produced no sequences -- check the data paths")
        print(RULE)
        print("Ground-truth vs predicted step streams (same sequences, same order model)")
        print(RULE)
        print(f"  sequences compared        : {s['n']}  ({args.fold} fold, view {args.view})")
        print(f"  steps  GT / predicted     : {s['gt_transitions']} / {s['pred_transitions']}")
        print(f"  flags  GT / predicted     : {s['gt_flags']} / {s['pred_flags']}")
        print(f"  per-transition flag rate  : GT {s['gt_fpr']:.1f}%   "
              f"predicted {s['pred_fpr']:.1f}%")
        print(f"  sequences with >=1 flag   : GT {s['gt_seq_flagged']:.0f}%   "
              f"predicted {s['pred_seq_flagged']:.0f}%")
        print(RULE)
        print("  Reading this: the annotation-only protocol in `mistake_eval` scores GT")
        print("  streams, so it measures the ORDER MODEL alone. The predicted column is")
        print("  the whole pipeline, where recogniser errors (oscillating/repeated steps)")
        print("  also fire the monitor. The gap is the cost of recogniser noise, and it")
        print("  is why the deployed system needs step smoothing, not a looser order model.")
        print(RULE)
        return 0

    # --- resolve the stream source, degrading gracefully ---
    ann_ok = os.path.isdir(os.path.join(args.data_root, "coarse-annotations"))
    ckpt_ok = os.path.isfile(args.ckpt)
    source = args.source
    if source == "auto":
        source = "model" if (ann_ok and ckpt_ok) else ("gt" if ann_ok else "sample")
    if source == "model" and not (ann_ok and ckpt_ok):
        raise SystemExit(f"--source model needs annotations ({ann_ok=}) and a checkpoint "
                         f"at {args.ckpt} ({ckpt_ok=})")
    if source == "gt" and not ann_ok:
        raise SystemExit(f"--source gt needs the coarse annotations under {args.data_root}")
    if source == "sample" and antic is None:
        raise SystemExit("--source sample needs an anticipation model "
                         "(python -m phase3_activity.tas.anticipation)")

    if source == "model":
        try:
            name, steps = stream_from_model(args.data_root, args.features_root, args.view,
                                            args.fold, args.procedure, args.ckpt,
                                            args.index, args.name)
        except SystemExit:
            if args.source != "auto":
                raise
            print("[info] features unavailable -- falling back to the ground-truth stream")
            source = "gt"
            name, steps = stream_from_gt(args.data_root, args.fold, args.procedure,
                                         args.index, args.name)
    elif source == "gt":
        name, steps = stream_from_gt(args.data_root, args.fold, args.procedure,
                                     args.index, args.name)
    else:
        name, steps = stream_sampled(antic)

    swapped, fault_at, fault_how = None, -1, ""
    if args.inject_fault:
        steps, swapped, fault_at, fault_how = inject_fault(proc, steps, args.seed)
        if swapped is None:
            print("[warn] this sequence contains no pair of steps the order model has a "
                  "learned constraint for,")
            print("       so no fault could be planted. Try --index 1 (a different "
                  "sequence), or")
            print("       --source sample, which generates a clean stream that is always "
                  "injectable.")

    described = {"model": f"TRAINED model predictions ({args.view})",
                 "gt": f"ground-truth annotations ({args.fold} fold)",
                 "sample": "learned-model walk (NO data -- mechanism demo only)"}[source]

    print(RULE)
    print("Phase 3 -- dynamic workflow monitor")
    print(RULE)
    print(f"  sequence   : {name}")
    print(f"  source     : {described}")
    print(f"  order model: {len(proc.constraint_stats)} learned precedence constraints")
    print(f"  anticipate : {'%d steps' % len(antic.vocab) if antic else 'disabled (no model)'}")
    if swapped is not None:
        a, b = swapped
        where = f"at step {fault_at + 1}"
        if fault_how == "swapped":
            print(f"  INJECTED FAULT: swapped '{proc.name(a)}' and '{proc.name(b)}' "
                  f"({where}) -- '{proc.name(a)}'")
        else:
            print(f"  INJECTED FAULT: inserted an extra '{proc.name(a)}' {where}, "
                  f"after '{proc.name(b)}' -- '{proc.name(a)}'")
        print(f"                  must come first, so this build is now out of order.")
    print(RULE)
    print("  step-by-step replay ('<-#1' = the step was the top-1 prediction made "
          "before\n  it happened; a MISTAKE line = the order model flagged it):")
    print()

    # In `sample` mode the stream is generated BY the anticipation model, so scoring it
    # would be circular -- show the mechanism, report no accuracy.
    summary = replay(steps, proc, antic, k=args.topk, score=(source != "sample"))

    print()
    print(RULE)
    n_ev = len(summary["events"])
    print(f"  steps replayed      : {summary['n_steps']}")
    print(f"  mistakes flagged    : {n_ev}"
          + (f"  -> {', '.join(sorted({e.kind for _i, e in summary['events']}))}" if n_ev else ""))
    if summary["top1"] is not None:
        print(f"  anticipation (this sequence): top-1 {summary['top1']:.0f}%  "
              f"top-{summary['k']} {summary['topk']:.0f}%")
        print("  (single-sequence numbers are noisy -- the fold-level result is "
              "top-1 15.5% / top-3 27.5%,")
        print("   see `python -m phase3_activity.tas.anticipation`)")
    else:
        print("  anticipation        : not scored in 'sample' mode (would be circular)")
    if args.inject_fault:
        if swapped is None:
            # Always print the line the launcher tells people to look for. Silence here is
            # indistinguishable from a broken detector.
            print("  injected fault      : NOT INJECTED (see the warning above)")
        else:
            # Match on the POSITION, not just the step id. A recognised stream often
            # already violates order using the very same step, so "did any event mention
            # step a" would report CAUGHT even with nothing injected.
            caught = any(i == fault_at and e.kind == "order_violation"
                         for i, e in summary["events"])
            print(f"  injected fault      : {'CAUGHT' if caught else 'MISSED'} "
                  f"(at step {fault_at + 1}, {fault_how})")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

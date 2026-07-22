"""Next-action (next-step) anticipation for Assembly101 step recognition.

The supervisor's staged plan puts anticipation last, after step recognition and
mistake detection. The *canonical* Assembly101 anticipation benchmark predicts the
**fine-grained** action (1380 classes) at a future time t+Δ, which needs the
fine-grained annotations — a separate gated download we don't have. As with mistake
detection, we build the version that runs on the coarse annotations already on disk
and maps directly onto the AR use case:

    given the steps performed so far, predict the NEXT step ("next: attach cabin").

This is a sequence-prediction problem over the coarse step vocabulary. The model is a
first-order Markov transition model with three workflow-aware refinements that a raw
bigram lacks:

  * **Done-masking.** In assembly each step is performed roughly once, so a step
    already completed is a poor "next" guess — completed steps are excluded.
  * **Precedence feasibility.** A step whose hard prerequisites (from the learned
    `ProcedureModel`) have not all been performed cannot come next — it is masked out.
  * **Cold-start.** With no history yet, the learned start-step distribution predicts
    the first step.

Evaluated honestly against two baselines (marginal frequency, raw bigram) with proper
train/val separation. Pure-Python (no torch/lmdb) and self-contained, so it needs only
the small coarse annotations and unit-tests without data.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class AnticipationModel:
    """Next-step predictor learned from training step sequences."""
    alpha: float = 0.1                        # add-alpha smoothing on transitions
    n_videos: int = 0
    vocab: set = field(default_factory=set)
    unigram: Counter = field(default_factory=Counter)          # step occurrence counts
    starts: Counter = field(default_factory=Counter)           # first-step counts
    bigram: Dict[Tuple[int, int], int] = field(default_factory=Counter)   # (a,b) -> count
    rowsum: Counter = field(default_factory=Counter)           # sum_b bigram[(a,b)]
    prereqs: Dict[int, set] = field(default_factory=lambda: defaultdict(set))  # feasibility
    id_to_action: Dict[int, str] = field(default_factory=dict)

    # -- fitting --
    @classmethod
    def fit(cls, sequences: Sequence[Sequence[int]], alpha: float = 0.1,
            procedure: Optional[object] = None,
            id_to_action: Optional[Dict[int, str]] = None) -> "AnticipationModel":
        m = cls(alpha=float(alpha), id_to_action=dict(id_to_action or {}))
        for seq in sequences:
            seq = list(seq)
            if not seq:
                continue
            m.n_videos += 1
            m.starts[seq[0]] += 1
            for s in seq:
                m.vocab.add(s)
                m.unigram[s] += 1
            for a, b in zip(seq, seq[1:]):
                m.bigram[(a, b)] += 1
                m.rowsum[a] += 1
        if procedure is not None:                              # copy feasibility constraints
            for step in getattr(procedure, "vocab", set()):
                m.prereqs[step] = set(procedure.prerequisites(step))
            m.vocab |= set(getattr(procedure, "vocab", set()))
        return m

    LAMBDA = 0.7        # transition weight in the back-off interpolation (rest = marginal)

    def name(self, step: int) -> str:
        return self.id_to_action.get(int(step), f"cls-{int(step)}")

    # -- scoring (all predictors exclude already-completed steps) --
    def _dist_marginal(self, done: set) -> Dict[int, float]:
        sc = {n: self.unigram.get(n, 0) + self.alpha for n in self.vocab if n not in done}
        z = sum(sc.values()) or 1.0
        return {n: v / z for n, v in sc.items()}

    def _dist_bigram(self, current: Optional[int], done: set) -> Dict[int, float]:
        V = max(len(self.vocab), 1)
        sc: Dict[int, float] = {}
        for n in self.vocab:
            if n in done:
                continue
            if current is None:
                sc[n] = (self.starts.get(n, 0) + self.alpha) / (self.n_videos + self.alpha * V)
            else:
                sc[n] = (self.bigram.get((current, n), 0) + self.alpha) / \
                        (self.rowsum.get(current, 0) + self.alpha * V)
        z = sum(sc.values()) or 1.0
        return {n: v / z for n, v in sc.items()}

    @staticmethod
    def _topk(scores: Dict[int, float], k: int) -> List[Tuple[int, float]]:
        # sort by descending prob, then id, for deterministic ties
        return sorted(scores.items(), key=lambda t: (-t[1], t[0]))[:k]

    def predict_next(self, history: Sequence[int], k: int = 3,
                     use_feasibility: bool = False) -> List[Tuple[int, float]]:
        """Top-k (step_id, probability) for the next step given the ordered history.

        The shipped model: back-off interpolation of the transition distribution with
        the marginal, done-masked. `use_feasibility` (off by default) additionally
        drops steps whose learned prerequisites are unmet — it HURTS here because
        assembly ordering is loose, so it is opt-in only."""
        done = set(history)
        current = history[-1] if len(history) else None
        pm = self._dist_marginal(done)
        pb = self._dist_bigram(current, done)
        scores = {n: self.LAMBDA * pb.get(n, 0.0) + (1 - self.LAMBDA) * pm.get(n, 0.0)
                  for n in pb}
        if use_feasibility:
            feas = {n: s for n, s in scores.items()
                    if not self.prereqs.get(n) or self.prereqs[n] <= done}
            scores = feas or scores                            # fall back if all masked
        return self._topk(scores, k)

    def predict_bigram(self, history: Sequence[int], k: int = 3) -> List[Tuple[int, float]]:
        """Baseline: pure transition model, done-masked (no marginal back-off)."""
        done = set(history)
        current = history[-1] if len(history) else None
        return self._topk(self._dist_bigram(current, done), k)

    def predict_marginal(self, history: Sequence[int], k: int = 3) -> List[Tuple[int, float]]:
        """Baseline: global step frequency, done-masked, NO sequential information."""
        return self._topk(self._dist_marginal(set(history)), k)

    def next_step_names(self, history: Sequence[int], k: int = 3) -> List[str]:
        return [self.name(n) for n, _p in self.predict_next(history, k)]

    # -- (de)serialisation --
    def to_dict(self) -> dict:
        return {
            "version": 1, "alpha": self.alpha, "n_videos": self.n_videos,
            "vocab": sorted(int(s) for s in self.vocab),
            "unigram": {str(int(k)): int(v) for k, v in self.unigram.items()},
            "starts": {str(int(k)): int(v) for k, v in self.starts.items()},
            "bigram": [[int(a), int(b), int(c)] for (a, b), c in self.bigram.items()],
            "prereqs": [[int(b), sorted(int(a) for a in pre)]
                        for b, pre in self.prereqs.items() if pre],
            "id_to_action": {str(int(k)): v for k, v in self.id_to_action.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnticipationModel":
        m = cls(alpha=float(d.get("alpha", 0.1)), n_videos=int(d.get("n_videos", 0)))
        m.vocab = set(int(s) for s in d.get("vocab", []))
        m.unigram = Counter({int(k): int(v) for k, v in d.get("unigram", {}).items()})
        m.starts = Counter({int(k): int(v) for k, v in d.get("starts", {}).items()})
        m.bigram = Counter()
        m.rowsum = Counter()
        for a, b, c in d.get("bigram", []):
            m.bigram[(int(a), int(b))] = int(c)
            m.rowsum[int(a)] += int(c)
        m.prereqs = defaultdict(set)
        for b, pre in d.get("prereqs", []):
            m.prereqs[int(b)] = set(int(a) for a in pre)
        m.id_to_action = {int(k): v for k, v in d.get("id_to_action", {}).items()}
        return m

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "AnticipationModel":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# --- evaluation ---------------------------------------------------------------
def evaluate(model: AnticipationModel, val_seqs: Sequence[Tuple[str, List[int]]],
             ks: Sequence[int] = (1, 3)) -> dict:
    """Walk every val sequence; at each position predict the step from the history
    before it (position 0 = cold start). Report top-k accuracy for the full model and
    the marginal / bigram baselines. Higher is better."""
    kmax = max(ks)
    tot = 0
    hits = {name: {k: 0 for k in ks} for name in ("marginal", "bigram", "full")}
    predictors = {
        "marginal": lambda h: model.predict_marginal(h, kmax),
        "bigram": lambda h: model.predict_bigram(h, kmax),
        "full": lambda h: model.predict_next(h, kmax),
    }
    for _name, seq in val_seqs:
        for i in range(len(seq)):
            history, true = seq[:i], seq[i]
            tot += 1
            for pname, fn in predictors.items():
                top = [n for n, _p in fn(history)]
                for k in ks:
                    if true in top[:k]:
                        hits[pname][k] += 1
    acc = {p: {k: 100.0 * hits[p][k] / max(tot, 1) for k in ks} for p in hits}
    return {"n_predictions": tot, "ks": tuple(ks), "accuracy": acc}


# --- build + eval CLI ---------------------------------------------------------
def main(argv=None) -> int:
    import argparse

    from .dataset import load_actions_csv
    from .procedure import ProcedureModel, load_step_sequences

    ap = argparse.ArgumentParser(
        description="Learn + evaluate a next-step anticipation model "
                    "(coarse annotations only — no features needed).")
    ap.add_argument("--data-root", default="phase3_activity/data")
    ap.add_argument("--procedure", default="assembly",
                    choices=["assembly", "disassembly", "both"])
    ap.add_argument("--alpha", type=float, default=0.1, help="transition smoothing")
    ap.add_argument("--min-support", type=int, default=20,
                    help="procedure-model support (for feasibility masking)")
    ap.add_argument("--precedence-tau", type=float, default=0.99)
    ap.add_argument("--out", default="phase3_activity/models/anticipation_model.json")
    ap.add_argument("--examples", type=int, default=0,
                    help="print this many worked next-step predictions from val")
    args = ap.parse_args(argv)

    ann = os.path.join(args.data_root, "coarse-annotations")
    actions_dict, id_to_action = load_actions_csv(os.path.join(ann, "actions.csv"))
    splits, labels = os.path.join(ann, "coarse_splits"), os.path.join(ann, "coarse_labels")
    train = load_step_sequences(splits, labels, "train", actions_dict, procedure=args.procedure)
    val = load_step_sequences(splits, labels, "val", actions_dict, procedure=args.procedure)
    if not train or not val:
        print("no step sequences found — check --data-root"); return 1

    # a procedure model provides the precedence feasibility mask
    proc = ProcedureModel.fit([s for _n, s in train], min_support=args.min_support,
                              precedence_tau=args.precedence_tau, id_to_action=id_to_action)
    model = AnticipationModel.fit([s for _n, s in train], alpha=args.alpha,
                                  procedure=proc, id_to_action=id_to_action)
    model.save(args.out)

    r = evaluate(model, val, ks=(1, 3))
    acc = r["accuracy"]
    print(f"procedure={args.procedure}  train videos={model.n_videos}  "
          f"distinct steps={len(model.vocab)}  val predictions={r['n_predictions']}")
    print(f"{'predictor':<10} {'top-1':>7} {'top-3':>7}")
    for p in ("marginal", "bigram", "full"):
        print(f"{p:<10} {acc[p][1]:>6.1f}% {acc[p][3]:>6.1f}%")
    print(f"saved model -> {args.out}")

    if args.examples > 0:
        print("\nworked next-step predictions (val):")
        shown = 0
        for _name, seq in val:
            for i in range(1, len(seq)):
                if shown >= args.examples:
                    break
                hist = seq[:i]
                top = model.predict_next(hist, k=3)
                truth = model.name(seq[i])
                preds = ", ".join(f"{model.name(n)} ({p*100:.0f}%)" for n, p in top)
                hit = "OK " if seq[i] in [n for n, _ in top] else "-- "
                print(f"  {hit}after '{model.name(hist[-1])}'  -> next: {preds}   (actual: {truth})")
                shown += 1
            if shown >= args.examples:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

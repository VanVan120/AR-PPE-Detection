"""Assembly-order mistake detection for Assembly101 step recognition.

The supervisor's staged plan puts **mistake detection** after step recognition. The
*canonical* Assembly101 mistake benchmark is defined on the FINE-GRAINED annotations
(each fine action tagged correct / mistake / correction), which are a separate gated
download. Those are not required here — and for the real target use case ("dynamic
workflow monitoring for inspection via AR glasses") a more directly useful signal is:

    learn the expected assembly ORDER from the training step sequences, then flag a
    recognised step stream that deviates from it.

That is exactly what this module does, using only the small coarse annotations (no
34 GB features), so it covers *every* training assembly and is fully runnable offline.

The core idea — robust to Assembly101's partially free ordering:

  * **Precedence constraints.** For each ordered pair (a, b) we measure, across the
    training videos containing both, how often `a` occurs before `b`. If that ratio is
    near-deterministic (>= `precedence_tau`) with enough support, `a` is a real
    *prerequisite* of `b` (e.g. "attach chassis" before "screw chassis"). Genuinely
    interchangeable steps sit near 50/50 and never become constraints, so they are
    never flagged. This is the primary, low-false-positive signal.
  * **Transition novelty (secondary, optional).** A step transition `a -> b` never seen
    in training is mildly surprising. Weaker and noisier, so it is separable and
    reported apart from precedence.

`MistakeMonitor` applies the model **online** (one step at a time), which is exactly
how the Phase 2 seam consumes the recognised-step stream: when a new step is entered,
if a hard prerequisite has not yet occurred this session, that is a mistake.

Everything here is pure-Python (no numpy/torch/lmdb), so it unit-tests without data.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# --- step-sequence extraction -------------------------------------------------
def reduce_segments_to_step_ids(segments: Sequence[Tuple[int, int, str]],
                                actions_dict: Dict[str, int]) -> List[int]:
    """A coarse_labels segment list -> the ordered list of step ids, with consecutive
    duplicates collapsed (a workflow is a sequence of *distinct adjacent* steps).
    Unknown classes are skipped."""
    steps: List[int] = []
    for _s, _e, cls in segments:
        if cls not in actions_dict:
            continue
        sid = actions_dict[cls]
        if not steps or steps[-1] != sid:
            steps.append(sid)
    return steps


def load_step_sequences(splits_dir: str, coarse_labels_dir: str, fold: str,
                        actions_dict: Dict[str, int],
                        procedure: str = "assembly") -> List[Tuple[str, List[int]]]:
    """Read a fold's coarse annotations (NO features/LMDB needed) into ordered step-id
    sequences: [(annotation_name, [step_id, ...]), ...]. Skips empty/missing files.

    `procedure` selects which recordings to keep: 'assembly' (default — the forward
    build workflow), 'disassembly' (the reverse), or 'both'. Assembly and disassembly
    are OPPOSITE procedures, so pooling them corrupts the learned order — keep them
    apart unless you explicitly want both."""
    from .dataset import parse_coarse_label_file, video_ids_for_fold
    names = video_ids_for_fold(splits_dir, fold)
    if procedure == "assembly":
        names = [n for n in names if n.startswith("assembly_")]
    elif procedure == "disassembly":
        names = [n for n in names if n.startswith("disassembly_")]
    elif procedure != "both":
        raise ValueError(f"unknown procedure '{procedure}' (assembly | disassembly | both)")
    out: List[Tuple[str, List[int]]] = []
    for name in names:
        path = os.path.join(coarse_labels_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            segments = parse_coarse_label_file(fh.read())
        steps = reduce_segments_to_step_ids(segments, actions_dict)
        if steps:
            out.append((name, steps))
    return out


# --- the learned procedure model ---------------------------------------------
@dataclass
class ProcedureModel:
    """An expected-workflow model learned from training step sequences.

    Build with `ProcedureModel.fit(sequences, ...)`. Serialises to a compact,
    self-describing JSON (`save`/`load`) that carries the runtime-relevant parts:
    the prerequisite constraints, the seen transitions, the start steps, the
    vocabulary and (optionally) an id -> action-name map for readable reports.
    """
    min_support: int = 20
    precedence_tau: float = 0.99
    n_videos: int = 0
    vocab: set = field(default_factory=set)
    step_count: Counter = field(default_factory=Counter)
    starts: set = field(default_factory=set)
    transitions: set = field(default_factory=set)                 # seen (a, b) bigrams
    # a must precede b  ->  {b: {a, ...}}
    prereqs: Dict[int, set] = field(default_factory=lambda: defaultdict(set))
    # (a, b) -> (support, before_ratio) for the constraints that were kept (reporting)
    constraint_stats: Dict[Tuple[int, int], Tuple[int, float]] = field(default_factory=dict)
    id_to_action: Dict[int, str] = field(default_factory=dict)

    # -- fitting --
    @classmethod
    def fit(cls, sequences: Sequence[Sequence[int]], min_support: int = 20,
            precedence_tau: float = 0.99,
            id_to_action: Optional[Dict[int, str]] = None) -> "ProcedureModel":
        m = cls(min_support=int(min_support), precedence_tau=float(precedence_tau),
                id_to_action=dict(id_to_action or {}))
        cooccur: Dict[Tuple[int, int], int] = Counter()   # unordered pair, stored a<b
        before: Dict[Tuple[int, int], int] = Counter()    # times a occurred before b
        for seq in sequences:
            seq = list(seq)
            if not seq:
                continue
            m.n_videos += 1
            m.starts.add(seq[0])
            distinct = list(dict.fromkeys(seq))            # first-occurrence order
            first_pos = {s: i for i, s in enumerate(seq)}  # first occurrence index
            for s in distinct:
                m.vocab.add(s)
                m.step_count[s] += 1
            for a, b in zip(seq, seq[1:]):
                m.transitions.add((a, b))
            # pairwise precedence over the distinct step set of this video
            for i in range(len(distinct)):
                for j in range(i + 1, len(distinct)):
                    a, b = distinct[i], distinct[j]
                    key = (a, b) if a < b else (b, a)
                    cooccur[key] += 1
                    lo, hi = key
                    if first_pos[lo] < first_pos[hi]:
                        before[key] += 1
        # distil precedence constraints
        for (lo, hi), support in cooccur.items():
            if support < m.min_support:
                continue
            r = before[(lo, hi)] / support                 # P(lo before hi)
            if r >= m.precedence_tau:                       # lo almost always precedes hi
                m.prereqs[hi].add(lo)
                m.constraint_stats[(lo, hi)] = (support, r)
            elif (1.0 - r) >= m.precedence_tau:             # hi almost always precedes lo
                m.prereqs[lo].add(hi)
                m.constraint_stats[(hi, lo)] = (support, 1.0 - r)
        return m

    def prerequisites(self, step: int) -> set:
        """Steps that must precede `step` (a in a -> step)."""
        return set(self.prereqs.get(step, set()))

    def successors(self, step: int) -> set:
        """Steps that `step` must precede (b in step -> b). Built from the constraints;
        used for online order-violation detection."""
        return {b for (a, b) in self.constraint_stats if a == step}

    def constraints(self) -> List[Tuple[int, int, int, float]]:
        """[(a, b, support, ratio), ...] meaning a must precede b — sorted, strongest
        support first, for readable reporting."""
        rows = [(a, b, s, r) for (a, b), (s, r) in self.constraint_stats.items()]
        rows.sort(key=lambda t: (-t[2], -t[3]))
        return rows

    def name(self, step: int) -> str:
        return self.id_to_action.get(int(step), f"cls-{int(step)}")

    # -- (de)serialisation --
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "params": {"min_support": self.min_support, "precedence_tau": self.precedence_tau},
            "n_videos": self.n_videos,
            "vocab": sorted(int(s) for s in self.vocab),
            "step_count": {str(int(k)): int(v) for k, v in self.step_count.items()},
            "starts": sorted(int(s) for s in self.starts),
            "transitions": sorted([int(a), int(b)] for a, b in self.transitions),
            "constraints": [[int(a), int(b), int(s), round(float(r), 4)]
                            for (a, b, s, r) in self.constraints()],
            "id_to_action": {str(int(k)): v for k, v in self.id_to_action.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProcedureModel":
        p = d.get("params", {})
        m = cls(min_support=int(p.get("min_support", 20)),
                precedence_tau=float(p.get("precedence_tau", 0.99)),
                n_videos=int(d.get("n_videos", 0)))
        m.vocab = set(int(s) for s in d.get("vocab", []))
        m.step_count = Counter({int(k): int(v) for k, v in d.get("step_count", {}).items()})
        m.starts = set(int(s) for s in d.get("starts", []))
        m.transitions = set((int(a), int(b)) for a, b in d.get("transitions", []))
        m.id_to_action = {int(k): v for k, v in d.get("id_to_action", {}).items()}
        m.prereqs = defaultdict(set)
        for row in d.get("constraints", []):
            a, b, s, r = int(row[0]), int(row[1]), int(row[2]), float(row[3])
            m.prereqs[b].add(a)
            m.constraint_stats[(a, b)] = (s, r)
        return m

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProcedureModel":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# --- online mistake detection -------------------------------------------------
@dataclass
class MistakeEvent:
    step: int                       # the step whose entry triggered the flag
    kind: str                       # 'order_violation' | 'novel_transition' | 'unknown_step'
    prev: Optional[int]             # the step we transitioned from (None at sequence start)
    violated: List[int] = field(default_factory=list)   # partner steps in the order rule
    detail: str = ""                # human-readable reason
    severity: str = "warning"       # reserved for future severity grading


class MistakeMonitor:
    """Replays a recognised-step stream against a `ProcedureModel`, one step at a time.

    Order-violation rule (the key to a low false-positive rate): a constraint `a -> b`
    (a must precede b) is only *violated* when BOTH steps are actually performed and in
    the wrong order. That is detectable online without knowing the future: when a step
    `s` is entered, if some step `b` that `s` must precede (`s -> b`) has ALREADY been
    seen, then `s` came after `b` — a real reordering. A step that is simply absent
    (part of a different assembly) never triggers this, so absent steps are not
    mistaken for skipped prerequisites.

    `observe(step)` fires at most one `MistakeEvent` per *transition* (a change of
    step), in priority order: unknown_step > order_violation > novel_transition.
    Repeated per-frame/per-clip reports of the same step are naturally de-duplicated.
    Call `reset()` between independent sequences.
    """

    def __init__(self, model: ProcedureModel, use_novel_transition: bool = False):
        self.model = model
        self.use_novel_transition = bool(use_novel_transition)
        # precompute the successor map (s -> {b : s must precede b}) once
        self._succ: Dict[int, set] = defaultdict(set)
        for (a, b) in model.constraint_stats:
            self._succ[a].add(b)
        self.reset()

    def reset(self) -> None:
        self.seen: set = set()
        self.prev: Optional[int] = None

    def observe(self, step: int) -> Optional[MistakeEvent]:
        step = int(step)
        if step == self.prev:
            return None                                   # same step continuing — no transition
        event: Optional[MistakeEvent] = None
        m = self.model
        if m.vocab and step not in m.vocab:
            event = MistakeEvent(step, "unknown_step", self.prev,
                                 detail=f"'{m.name(step)}' never seen in training")
        else:
            # steps that `step` must precede but that already happened -> out of order
            violated = sorted(self._succ.get(step, set()) & self.seen)
            if violated:
                names = ", ".join(m.name(x) for x in violated)
                event = MistakeEvent(step, "order_violation", self.prev, violated=violated,
                                     detail=f"'{m.name(step)}' should precede: {names} "
                                            "(which already occurred)")
            elif (self.use_novel_transition and self.prev is not None
                  and (self.prev, step) not in m.transitions):
                event = MistakeEvent(step, "novel_transition", self.prev,
                                     detail=f"'{m.name(self.prev)}' -> '{m.name(step)}' "
                                            "not seen in training")
        self.seen.add(step)
        self.prev = step
        return event


def score_sequence(model: ProcedureModel, steps: Sequence[int],
                   use_novel_transition: bool = False) -> List[Tuple[int, MistakeEvent]]:
    """Replay a whole step sequence through a fresh monitor; return (index, event) for
    every flagged transition."""
    mon = MistakeMonitor(model, use_novel_transition=use_novel_transition)
    out: List[Tuple[int, MistakeEvent]] = []
    for i, s in enumerate(steps):
        ev = mon.observe(s)
        if ev is not None:
            out.append((i, ev))
    return out


# --- build CLI ----------------------------------------------------------------
def main(argv=None) -> int:
    import argparse

    from .dataset import load_actions_csv

    ap = argparse.ArgumentParser(
        description="Learn an expected-assembly-order model for mistake detection "
                    "(uses only the coarse annotations — no features needed).")
    ap.add_argument("--data-root", default="phase3_activity/data")
    ap.add_argument("--fold", default="train", choices=["train", "train_val", "val"],
                    help="fold to learn the model from (default: train — never val, "
                         "which is held out for evaluation)")
    ap.add_argument("--procedure", default="assembly",
                    choices=["assembly", "disassembly", "both"],
                    help="which workflow to model (default: assembly)")
    ap.add_argument("--min-support", type=int, default=20,
                    help="min #videos a step-pair must co-occur in to form a constraint")
    ap.add_argument("--precedence-tau", type=float, default=0.99,
                    help="how deterministic an ordering must be to count as a prerequisite")
    ap.add_argument("--out", default="phase3_activity/models/procedure_model.json")
    ap.add_argument("--top", type=int, default=15, help="how many constraints to print")
    args = ap.parse_args(argv)

    ann = os.path.join(args.data_root, "coarse-annotations")
    actions_dict, id_to_action = load_actions_csv(os.path.join(ann, "actions.csv"))
    seqs = load_step_sequences(os.path.join(ann, "coarse_splits"),
                               os.path.join(ann, "coarse_labels"), args.fold, actions_dict,
                               procedure=args.procedure)
    if not seqs:
        print("no step sequences found — check --data-root"); return 1
    model = ProcedureModel.fit([s for _n, s in seqs], min_support=args.min_support,
                               precedence_tau=args.precedence_tau, id_to_action=id_to_action)
    model.save(args.out)

    cons = model.constraints()
    lengths = [len(s) for _n, s in seqs]
    print(f"fold={args.fold}  procedure={args.procedure}  videos={model.n_videos}  "
          f"distinct steps={len(model.vocab)}  avg len={sum(lengths) / len(lengths):.1f}")
    print(f"learned {len(cons)} precedence constraints "
          f"(min_support={args.min_support}, tau={args.precedence_tau})")
    print(f"top {min(args.top, len(cons))} by support:")
    for a, b, s, r in cons[:args.top]:
        print(f"  '{model.name(a)}'  ->  '{model.name(b)}'   "
              f"(support {s}, {r * 100:.0f}% of co-occurrences)")
    print(f"saved model -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

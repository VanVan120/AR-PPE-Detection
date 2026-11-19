"""Revision R1: shorten a training log for the repository without losing any message.

ultralytics redraws its progress bar many times per epoch, and in a log file every redraw is kept,
so a seed's log grows to tens of MB. This keeps every line that is not a progress-bar redraw,
and of each bar only its final state: the 100 % line or, if the bar was cut off (a crash), the
last line drawn before the next message. ANSI escape codes are removed. The full log stays on
disk next to it.

    .venv\\Scripts\\python.exe tools\\revision\\condense_log.py paper_materials\\revision_r1\\partA\\logs\\seed1.log
    -> paper_materials\\revision_r1\\partA\\logs\\seed1.condensed.log
"""
from __future__ import annotations

import argparse
import re

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BAR = re.compile(r"\s(\d{1,3})% \S* ?\d+/\d+")


def condense(text: str) -> str:
    out, pending = [], None

    def emit(s):
        if not out or out[-1] != s:
            out.append(s)

    for raw in text.split("\n"):
        for s in raw.split("\r"):
            s = ANSI.sub("", s).rstrip()
            if not s.strip():
                continue
            m = BAR.search(s)
            if m:
                if int(m.group(1)) == 100:
                    emit(s)
                    pending = None
                else:
                    pending = s
                continue
            if pending is not None:
                emit(pending)
                pending = None
            emit(s)
    if pending is not None:
        emit(pending)
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--out", help="default: <log without .log>.condensed.log")
    a = ap.parse_args(argv)
    out = a.out or re.sub(r"\.log$", "", a.log) + ".condensed.log"
    text = open(a.log, encoding="utf-8", errors="replace").read()
    res = condense(text)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(res)
    print(f"{a.log}: {len(text):,} chars -> {out}: {len(res):,} chars, {res.count(chr(10)):,} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

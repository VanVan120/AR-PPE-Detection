#!/usr/bin/env bash
# Revision R1, Part D: every tracker run on the cached MOT17 detections, then TrackEval.
# Two streams run side by side: the phase2 trackers (.venv, needs supervision) and the
# ultralytics/boxmot trackers (.venv-eval). Each run writes MOT files under
# data/revision_r1_cache/partD/trackers and a record under paper_materials/revision_r1/partD/runs.
#   bash tools/revision/run_partD.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 YOLO_AUTOINSTALL=false
L=paper_materials/revision_r1/partD/logs
mkdir -p "$L"

run() {   # env tracker det rate feed threads
  local py=$1 trk=$2 det=$3 rate=$4 feed=$5 thr=$6
  local tag="${det}_${trk}_${rate}_${feed}"
  local t0=$(date +%s)
  OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr "$py" -B tools/revision/mot17_track.py \
      --tracker "$trk" --det "$det" --rate "$rate" --feed "$feed" --threads "$thr" > "$L/track_$tag.log" 2>&1
  echo "$(date +%H:%M:%S) $tag exit $? in $(( $(date +%s) - t0 )) s"
}

phase2_stream() {
  for det in grouped coco; do
    for trk in bytetrack_phase2 workid_phase2; do
      run .venv/Scripts/python.exe $trk $det native pipeline 1
      run .venv/Scripts/python.exe $trk $det 7hz pipeline 1
    done
  done
  # sensitivity: the phase2 trackers fed every cached box (>= 0.10), not just > 0.35
  for det in grouped coco; do
    for trk in bytetrack_phase2 workid_phase2; do
      run .venv/Scripts/python.exe $trk $det native all 1
    done
  done
}

external_stream() {
  for det in grouped coco; do
    run .venv-eval/Scripts/python.exe botsort_reid $det native pipeline 2
    run .venv-eval/Scripts/python.exe strongsort $det native pipeline 3
    run .venv-eval/Scripts/python.exe deepocsort $det native pipeline 3
  done
}

phase2_stream > "$L/stream_phase2.log" 2>&1 &
p1=$!
external_stream > "$L/stream_external.log" 2>&1 &
p2=$!
wait $p1 $p2
cat "$L/stream_phase2.log" "$L/stream_external.log"
t0=$(date +%s)
OMP_NUM_THREADS=1 .venv-eval/Scripts/python.exe tools/revision/mot17_eval.py > "$L/eval.log" 2>&1
echo "$(date +%H:%M:%S) mot17_eval exit $? in $(( $(date +%s) - t0 )) s"

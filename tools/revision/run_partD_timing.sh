#!/usr/bin/env bash
# Revision R1, Part D: comparable CPU time per frame for every tracker.
# One run at a time, every tracker with 1 torch thread and 1 OpenCV thread, on the COCO
# detections (the larger box counts) for MOT17-02 (static, 1920x1080), MOT17-05 (moving,
# 640x480) and MOT17-11 (moving, 1920x1080). Timing covers the tracker call only (detection is
# cached; image decoding excluded). BoT-SORT's camera-motion compensation is also timed on its
# own. StrongSORT and Deep OC-SORT are timed twice: on the whole cache (their ReID embeds every
# input box) and pre-filtered at their own thresholds (min_conf 0.6; det_thresh 0.5), which gives
# identical tracks. Outputs go to data/ (scratch); the records are copied to partD/timing/.
# The GPU training job runs alongside, so absolute times carry that load.
set -u
cd "$(dirname "$0")/../.." || exit 1
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 YOLO_AUTOINSTALL=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SMOKE=data/revision_r1_cache/partD/timing
OUTT=paper_materials/revision_r1/partD/timing
L=paper_materials/revision_r1/partD/logs
mkdir -p "$SMOKE" "$OUTT" "$L"
SEQS="MOT17-02-FRCNN MOT17-05-FRCNN MOT17-11-FRCNN"

t() {   # env tracker [min_conf] -> one timing run into its own folder
  local py=$1 trk=$2 mc=${3:-}
  local tag="coco_${trk}${mc:+_prefilt$mc}"
  local extra=()
  [ -n "$mc" ] && extra=(--timing-min-conf "$mc")
  local t0=$(date +%s)
  "$py" -B tools/revision/mot17_track.py --tracker "$trk" --det coco --rate native --threads 1 \
      --cv-threads 1 --seqs $SEQS --smoke-dir "$SMOKE/$tag" "${extra[@]}" > "$L/timing_$tag.log" 2>&1
  echo "$(date +%H:%M:%S) $tag exit $? in $(( $(date +%s) - t0 )) s"
  cp "$SMOKE/$tag/runs/"*.json "$OUTT/$tag.json" 2>/dev/null
}

t .venv/Scripts/python.exe bytetrack_phase2
t .venv/Scripts/python.exe workid_phase2
t .venv-eval/Scripts/python.exe botsort_reid
t .venv-eval/Scripts/python.exe strongsort
t .venv-eval/Scripts/python.exe strongsort 0.6
t .venv-eval/Scripts/python.exe deepocsort
t .venv-eval/Scripts/python.exe deepocsort 0.5

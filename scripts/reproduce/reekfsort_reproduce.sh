#!/usr/bin/env bash
set -euo pipefail
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

# Reproduce the main ReEKF-SORT benchmark evaluations from cached detections.
# Run from the repository root.

boxmot generate --benchmark dancetrack --split val --detector yolox_x_dancetrack
boxmot eval --benchmark dancetrack --split val --tracker reekfsort --tracker-backend python --project runs/reekfsort_reproduce --name dancetrack_val --exist-ok

boxmot generate --benchmark mot17 --split ablation --detector yolox_x_mot17_ablation
boxmot eval --benchmark mot17 --split ablation --tracker reekfsort --tracker-backend python --project runs/reekfsort_reproduce --name mot17_ablation --exist-ok

boxmot generate --benchmark mot20 --split ablation --detector yolox_x_mot20_ablation
boxmot eval --benchmark mot20 --split ablation --tracker reekfsort --tracker-backend python --project runs/reekfsort_reproduce --name mot20_ablation --exist-ok

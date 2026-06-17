$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Run the full ReEKF-SORT ablation matrix on all paper benchmarks.
# Prepare data and detector weights first:
#   python scripts/reproduce/prepare_reekfsort_assets.py

python scripts/reproduce/reekfsort_ablation.py --benchmark dancetrack --split val
python scripts/reproduce/reekfsort_ablation.py --benchmark mot17 --split ablation
python scripts/reproduce/reekfsort_ablation.py --benchmark mot20 --split ablation

#!/bin/bash
source /home/users/myllykos/mambaforge/bin/activate advec

conda activate advec
cd /home/users/myllykos/mepsi/MEPS_advection_based_adjustment_of_radar_measurements
current_time=$(date -d "3 hours ago" "+%Y%m%d%H")$(printf "00")

python compute_advection_correction.py "$current_time"
python correct_radar_volume.py "$current_time"
python download_current_and_remove_oldest.py
python combine_the_latest_smartmet_files.py

#!/bin/bash
source /home/users/myllykos/mambaforge/bin/activate advec

conda activate advec
cd /home/users/myllykos/mepsi/MEPS_advection_based_adjustment_of_radar_measurements

python download_current_and_remove_oldest.py
python combine_the_latest_smartmet_files.py

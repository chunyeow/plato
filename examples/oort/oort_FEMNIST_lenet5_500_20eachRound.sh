#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=oort_FEMNIST_lenet5_1000_50eachRound_25least_no_test_stalenss_10_sleep10_2.out
filename='oort_FEMNIST_lenet5.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
python oort.py -c oort_FEMNIST_lenet5.yml -b /data/ykang/plato
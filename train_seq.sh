#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4 
#SBATCH -o out%j.txt
#SBATCH -e err%j.txt
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1

python train.py --model_name seq_model --split mini_data
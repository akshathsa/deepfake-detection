#!/bin/bash
#SBATCH -Jefficientvit
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t02:00:00
#SBATCH -o Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=arp9@gatech.edu
#SBATCH --partition=coc-gpu
#SBATCH -q coc-ice
cd $SLURM_SUBMIT_DIR

module load anaconda3/2022.05.0.1
conda activate deepfakes
python3 /home/hice1/arp9/scratch/deepfake-detection/models/vit/efficient-vit/train.py --workers=1 --max_train_videos=40 --max_val_videos=5 --config="/home/hice1/arp9/scratch/deepfake-detection/models/vit/efficient-vit/configs/architecture.yaml"
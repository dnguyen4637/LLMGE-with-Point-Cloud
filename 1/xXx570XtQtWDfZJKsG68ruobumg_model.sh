#!/bin/bash
#SBATCH --job-name=evaluateGene
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
echo "Launching Python Evaluation"
hostname

# Load GCC version 9.2.0
# module load gcc/13.2.0
module load cuda
module load anaconda3
# Activate Conda environment
conda activate llm_guided_env
export LD_LIBRARY_PATH=~/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# conda info

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Run Python script
PYTHONPATH=/home/hice1/dnguyen448/scratch/LLM-Guided-Evolution-Generic/sota/Pointnet_Pointnet2_pytorch/models/llmge_models:$PYTHONPATH python ./sota/Pointnet_Pointnet2_pytorch/train_classification.py --batch_size 216 --model "pointnet2_cls_ssg_xXx570XtQtWDfZJKsG68ruobumg" --data /storage/ice-shared/vip-vvk/data/llm_ge_data/modelnet40_normal_resampled --end_lr 0.001 --seed 21 --val_r 0.2 --amp --epoch 2

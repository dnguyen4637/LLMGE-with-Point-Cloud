#!/bin/bash
#SBATCH --job-name=llm_oper
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
echo "Launching AIsurBL"
hostname

# Load GCC version 9.2.0
# module load gcc/13.2.0
# module load cuda/11.8
module load cuda
module load anaconda3
# Activate Conda environment
conda activate llm_guided_env
export LD_LIBRARY_PATH=~/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# conda info

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Run Python script
python src/llm_mutation.py /home/hice1/dnguyen448/scratch/LLM-Guided-Evolution-Generic/sota/Pointnet_Pointnet2_pytorch/models/pointnet2_cls_ssg.py /home/hice1/dnguyen448/scratch/LLM-Guided-Evolution-Generic/sota/Pointnet_Pointnet2_pytorch/models/llmge_models/pointnet2_cls_ssg_xXxHXGqZZp9jodmMG41SmGAZK8h.py 1/xXxHXGqZZp9jodmMG41SmGAZK8h_model.txt --top_p 0.1 --temperature 0.2 --apply_quality_control 'False' --hugging_face False

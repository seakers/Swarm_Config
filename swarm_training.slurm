#!/bin/bash
#SBATCH --job-name=constellation_ppo
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12           # 12 CPUs for 8 parallel envs + overhead
#SBATCH --mem=90G                    # More RAM for 8 parallel envs
#SBATCH --gres=gpu:a100:2            # BOTH A100 40GB GPUs
#SBATCH --partition=gpu

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
ml GCCcore/13.2.0 Python/3.11.5
source .venv/bin/activate

# Critical: set OMP threads to 1 so PyTorch doesn't fight with multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Use local NVMe for fast temp I/O (checkpoints, logs during training)
export TMPDIR=/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR

cd $SLURM_SUBMIT_DIR
mkdir -p checkpoints logs

echo "Starting training..."
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU 0: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "GPU 1: $(python -c 'import torch; print(torch.cuda.get_device_name(1))')"

python main.py \
    --total_timesteps 5000000 \
    --rollout_steps 2048 \
    --max_episode_steps 1000 \
    --num_envs 8 \
    --num_workers 8 \
    --hidden_dim 256 \
    --num_gnn_layers 5 \
    --num_attention_heads 8 \
    --learning_rate 1e-4 \
    --use_amp \
    --save_dir "./checkpoints/${SLURM_JOB_ID}" \
    --log_dir "./logs/${SLURM_JOB_ID}"

echo "Job finished at: $(date)"

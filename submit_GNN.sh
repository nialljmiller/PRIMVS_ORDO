#!/bin/bash
#BATCH --job-name=PRIMVS_GNN
#SBATCH --output=logs/PRIMVS_GNN_%j.out
#SBATCH --error=logs/PRIMVS_GNN_%j.err
#SBATCH --partition=mb-a30
#SBATCH --account=galacticbulge
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64

# Print info for debugging
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

# Move to project directory
cd /project/galacticbulge/PRIMVS_ORDO || exit 1


# Activate your virtual environment
source ~/python_projects/venv/bin/activate

# Make sure logs dir exists
mkdir -p logs

# Run your Python script
python -u GNN.py

echo "Finished at: $(date)"


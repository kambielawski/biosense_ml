# BIOSENSE ML Engineer — Agent Memory

## VACC Access
- Host: `ktbielaw@login.vacc.uvm.edu`
- Remote project root: `~/projects/biosense_ml`
- VACC scripts (local): `/Users/kam/.claude/plugins/marketplaces/claude-plugins-official/plugins/vacc/skills/vacc/scripts/`
  - `vacc_session.sh` — ensure SSH tmux session named "vacc" is alive
  - `vacc_cmd.sh <cmd> [timeout]` — run a command in the vacc session
  - `vacc_status.sh [job_id]` — check squeue/sacct
  - `vacc_monitor.sh <job_id>` — poll until complete
  - `vacc_submit.sh` — submit helper (use `vacc_cmd.sh sbatch ...` directly if needed)
- Always run `vacc_session.sh` first; it prints SESSION_READY or SESSION_FAILED

## Repository Structure (local: `~/dev/biosense/biosense_ml`)
- `biosense_ml/models/` — model registry (ConvAutoencoder, etc.)
- `biosense_ml/utils/` — checkpointing, distributed, wandb logging
- `biosense_ml/pipeline/` — data discovery, preprocessing, WebDataset loaders
- `scripts/` — entry points: `train_autoencoder.py`, `train.py`, `preprocess.py`
- `slurm/` — Slurm scripts: `submit_autoencoder.sh`, `submit_train.sh`, etc.
- `configs/` — Hydra YAML configs (model, data, training, slurm)

## Key Model Detail
- `ConvAutoencoder` final activation: was `nn.Sigmoid()` (bug), now `nn.Identity()` as of commit `cea66e9`
- This fix is critical: ImageNet-normalized inputs span ~[-2.1, 2.6]; Sigmoid clips to [0,1], artificially bounds MSE loss

## Slurm Conventions
- Default autoencoder job: `sbatch slurm/submit_autoencoder.sh trainer.max_epochs=100`
- Partition: `nvgpu`, 1 GPU, 8 CPUs, 64GB, 24h wall time
- Preprocessing job: `sbatch --partition=general --cpus-per-task=16 --mem=64G --time=12:00:00 slurm/submit_preprocess.sh data=full data/preprocessing=resize data.biosense_archive_path=/users/k/k/kkannans/scratch/biosense_training_data data.processed_data_dir=/users/k/t/ktbielaw/projects/biosense_ml/data/processed`
  - `biosense_archive_path` and `processed_data_dir` are BOTH null in default config — must always be passed via CLI
  - Archive: `/users/k/k/kkannans/scratch/biosense_training_data` (another user's scratch, verify still accessible)
  - full.yaml lists 349 batch IDs but archive only has ~146 — "not found" warnings are normal
- Logs land in `~/projects/biosense_ml/logs/<jobid>_<jobname>.out/.err` (preprocess: `preprocess_<jobid>.out/.err`)
- Checkpoints in `~/projects/biosense_ml/outputs/autoencoder/checkpoints/`

## Project Notes Location
- Obsidian vault: `~/Documents/Universe/mortimer/projects/biosense-ml/biosense-ml.md`
- Daily journal: `~/Documents/Universe/mortimer/daily/YYYY-MM/YYYY-MM-DD.md`

## Experiment History (key jobs)
- Job 3259258: buggy Sigmoid run, ran to ~epoch 70, val_loss plateaued ~0.608 on node h2node05
- Job 3264101: first clean run post sigmoid-fix (cea66e9), 100 epochs, node dg-gpunode07, submitted 2026-03-05
- Job 3264115: full dataset re-preprocessing after crop fix (ec9668e), resize mode, 16 CPUs 64G 12h, node node410, submitted 2026-03-05

# BIOSENSE ML

ML research infrastructure for iterative model development on BIOSENSE data. Built with PyTorch, Hydra, wandb, and WebDataset. Designed for training on a Slurm-managed GPU cluster.

## Setup

```bash
# Create virtual environment and install dependencies
make setup
source .venv/bin/activate

# Configure wandb (one-time)
wandb login
```

## Data Pipeline

### Archive Location

The raw dataset lives in the BIOSENSE archive — a directory of `batch-NNNNNN/` subdirectories.
Point to it with `data.biosense_archive_path`:

```bash
data.biosense_archive_path="$HOME/BIOSENSE Dropbox/BIOSENSE Team Folder/biosense_communication_interface/prod/archive"
```

### Batches and Splits

Specify which batches to use via `data.batches`:

```bash
# Dev subset
python scripts/preprocess.py data.batches=[1,2,3,4,5,6,7,8,9] ...

# All batches
python scripts/preprocess.py 'data.batches=[1,2,3,...,349]' ...
```

Train/val/test splitting is controlled by ratios in the training config (`training.train_ratio`, `training.val_ratio`). The test set is the remainder. Defaults: 80/10/10.

### Preprocessing

Two preprocessing modes are available:

**Resize mode** (default): Downscale images and package into WebDataset tar shards.
```bash
python scripts/preprocess.py \
    data.biosense_archive_path=/path/to/archive \
    data.processed_data_dir=./data/processed \
    data.preprocessing.mode=resize \
    data.preprocessing.target_size=224
```

**Autoencoder mode**: Encode images into latent vectors and save as HDF5 (requires a pretrained autoencoder checkpoint).
```bash
python scripts/preprocess.py \
    data.preprocessing.mode=autoencoder \
    data.preprocessing.checkpoint_path=/path/to/encoder.pt
```

Preprocessing supports Slurm array jobs for parallelism — see `slurm/submit_preprocess.sh`.

### Processed Data Format

- **WebDataset**: Tar shards (`shard-000000.tar`, etc.) each containing `{key}.jpg` + `{key}.json` pairs.
- **HDF5**: A single `latents.h5` file with `latents`, `metadata`, and `keys` datasets.
- A `manifest.json` file records preprocessing config, sample count, and output paths.

## Training

### Local Training

```bash
# Train with default config
python scripts/train.py

# Override parameters from CLI
python scripts/train.py model.hidden_dim=256 training.optimizer.lr=3e-4 training.epochs=50
```

### Slurm Submission

**Via submitit (recommended):**
```bash
# Single run on Slurm
python scripts/train.py --multirun hydra/launcher=submitit_slurm

# Hyperparameter sweep on Slurm
python scripts/train.py --multirun hydra/launcher=submitit_slurm \
    training.optimizer.lr=1e-3,1e-4,1e-5 model.hidden_dim=128,256
```

**Via manual sbatch:**
```bash
sbatch slurm/submit_train.sh model=baseline training.epochs=200
```

### Resume from Checkpoint

```bash
python scripts/train.py resume_from=outputs/checkpoints/checkpoint_best.pt
```

## Experiment Tracking

All runs are logged to [Weights & Biases](https://wandb.ai). Each run records the full Hydra config, training/validation metrics, and learning rate schedule.

```bash
# View your runs
wandb ui
```

## Project Structure

```
configs/         Hydra YAML configs (model, data, training, slurm)
biosense_ml/pipeline/    Dataset classes, preprocessing, WebDataset/HDF5 utilities
biosense_ml/models/      Model definitions and registry
biosense_ml/training/    Training loop, metrics, checkpointing
biosense_ml/utils/       wandb helpers, distributed training, checkpoint management
scripts/         Entry points (preprocess, train, eval)
slurm/           Example Slurm batch scripts
tests/           Smoke tests
notebooks/       Data exploration notebooks
```

## Adding New Models

1. Create `biosense_ml/models/my_model.py` with a class that extends `nn.Module`.
2. Add a config file `configs/model/my_model.yaml`.
3. Register it in `biosense_ml/models/__init__.py`:
   ```python
   from biosense_ml.models.my_model import MyModel
   MODEL_REGISTRY["my_model"] = MyModel
   ```
4. Train with: `python scripts/train.py model=my_model`

## Testing

```bash
make test      # Run all tests
make lint      # Check code style
make format    # Auto-format code
```

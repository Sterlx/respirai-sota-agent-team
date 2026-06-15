# src/training/hpo.py
"""
Hyperparameter Optimization (HPO) with Optuna for ICBHI2017 Lung Sound Classification
=====================================================================================

Purpose
-------
This script runs an Optuna study to find the best hyperparameters for our
lung sound model on the official ICBHI2017 60/40 split. The objectives are:

- Maximise the ICBHI score (average of Sensitivity and Specificity per class).
- Use efficient search strategies (TPE sampler + Hyperband pruning).
- Save the best configuration to a YAML file for later training runs.

Why Optuna (not Ray Tune)?
---------------------------
- Single‑machine workflow (the team uses one training GPU at a time).
- Optuna has a very simple Python API, integrates natively with PyTorch,
  and supports pruning out of the box.
- No additional infrastructure (Ray cluster) required.
- Optuna’s TPE sampler often converges faster than random/grid search
  for small‑to‑medium search spaces.

Search space
------------
- learning_rate    : log‑uniform [1e-5, 1e-2]
- weight_decay     : log‑uniform [1e-6, 1e-2]
- batch_size       : categorical {16, 32, 64}
- n_mels           : categorical {64, 128, 256}
- dropout_rate     : uniform [0.1, 0.5] (used in classifier head)

Pruning
-------
We use HyperbandPruner: it allocates a budget (epochs) to each trial,
and prunes unpromising trials based on intermediate validation scores.
The training loop must call `trial.report(score, epoch)` after every epoch
and `trial.should_prune()` to terminate early.

Interface with training code
----------------------------
The objective function imports `run_training` from `src.training.train`.
That function must accept:
- A config dictionary containing the hyperparameters.
- An optional `optuna.trial.Trial` object for reporting.
It must return the final validation score (ICBHI score, higher is better).

If the module `src.training.train` is not yet implemented, you can replace
the `run_training` call with a dummy function for testing.

Output
------
- best_hparams.yaml  : the best hyperparameters found.
- (optional) SQLite database with study history if `--storage` is provided.

Usage
-----
    python src/training/hpo.py --study-name icbhi_v1 --n-trials 100

Skill references
----------------
# Skill source: external/AI-research-SKILLs/03-fine-tuning/axolotl/references/other.md
  (mixed precision & efficient training best practices)
# Skill source: external/AI-research-SKILLs/03-fine-tuning/unsloth/references/llms-txt.md
  (general fine‑tuning hyperparameter considerations)
"""

import argparse
import logging
import sys
from pathlib import Path

import optuna
import yaml
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

# Try to import the training function from the training engineer's module.
# If it fails, the script cannot proceed – the user is informed.
try:
    from src.training.train import run_training
except ImportError as e:
    logging.error("Could not import run_training from src.training.train. "
                  "Ensure train.py is implemented and on the Python path.")
    raise e

# ---------------------------------------------------------------------------
# Configuration of the HPO study
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "learning_rate":  (1e-5, 1e-2),   # log‑uniform
    "weight_decay":   (1e-6, 1e-2),   # log‑uniform
    "batch_size":     [16, 32, 64],    # categorical
    "n_mels":         [64, 128, 256],  # categorical
    "dropout_rate":   (0.1, 0.5),      # uniform
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Optuna HPO for ICBHI2017 lung sound model."
    )
    parser.add_argument("--study-name", type=str, default="icbhi_hpo",
                        help="Name of the Optuna study (used for storage and logging).")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of HPO trials to run.")
    parser.add_argument("--direction", type=str, default="maximize",
                        choices=["maximize", "minimize"],
                        help="Direction of optimisation (ICBHI score is to be maximised).")
    parser.add_argument("--storage", type=str, default=None,
                        help="SQLite database URL (e.g., sqlite:///hpo.db) for persistent storage. "
                             "If not given, an in‑memory study is used.")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory where best_hparams.yaml will be saved.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Objective function called by Optuna in each trial
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """
    Train a model with hyperparameters sampled by the trial,
    and return the ICBHI validation score (higher = better).

    Pruning is performed via trial.report() inside run_training().

    Returns
    -------
    float
        ICBHI score on the official validation set.
    """
    # Sample hyperparameters from the search space
    hparams = {
        "learning_rate": trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"], log=True),
        "weight_decay":  trial.suggest_float("weight_decay",  *SEARCH_SPACE["weight_decay"], log=True),
        "batch_size":    trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
        "n_mels":        trial.suggest_categorical("n_mels",     SEARCH_SPACE["n_mels"]),
        "dropout_rate":  trial.suggest_float("dropout_rate",     *SEARCH_SPACE["dropout_rate"]),
    }

    # The training function receives the trial object to enable
    # intermediate reporting and pruning.
    # It must internally call:
    #   trial.report(validation_score, step=epoch)
    #   if trial.should_prune():
    #       raise optuna.TrialPruned()
    trial.set_user_attr("hparams", hparams)

    val_score = run_training(hparams, trial=trial)
    return val_score

# ---------------------------------------------------------------------------
# Main HPO logic
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Set random seed for reproducibility
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep console clean
    sampler = TPESampler(seed=args.seed)
    pruner = HyperbandPruner(
        min_resource=1,           # minimum number of epochs before pruning is allowed
        max_resource=50,          # maximum number of epochs (budget)
        reduction_factor=3,       # typical setting for Hyperband
    )

    # Create or load the study
    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    logging.info(f"Starting Optuna study '{args.study_name}' with {args.n_trials} trials.")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Retrieve and save best hyperparameters
    best_trial = study.best_trial
    best_hparams = best_trial.params

    logging.info(f"Best trial #{best_trial.number}: score={best_trial.value:.4f}")
    logging.info("Best hyperparameters:")
    for k, v in best_hparams.items():
        logging.info(f"  {k}: {v}")

    output_path = Path(args.output_dir) / "best_hparams.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump({
            "study_name": args.study_name,
            "best_score": best_trial.value,
            "best_hyperparameters": best_hparams,
            "best_trial_number": best_trial.number,
        }, f, default_flow_style=False)

    logging.info(f"Best hyperparameters saved to {output_path}")

    # Optionally, write study statistics
    pruning_rate = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED) / len(study.trials)
    logging.info(f"Pruned trials: {pruning_rate:.1%}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

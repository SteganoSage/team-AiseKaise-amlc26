"""
Training module for the LightGBM binary classifier.

Trains a pair classifier: given features for an (S1, candidate) pair,
predict whether they refer to the same real-world business.

Key design decisions:
- Group-aware splits: by S1 entity, not by pair, to avoid leakage.
- Group K-fold out-of-fold (OOF) scores are used to pick the threshold and
  the number of boosting rounds; the final model is then trained on all pairs.
- Uses all S2/S3 records in the candidate pool for realistic difficulty.
- Hyperparameters from config.py for reproducibility.
"""

import os
import pickle

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

import config


def create_labels(pairs: list, ground_truth: dict) -> np.ndarray:
    """
    Create binary labels for candidate pairs.

    A pair (S1, candidate) is positive (1) if the candidate appears in the
    ground truth matched_entity_ids for that S1 entity.

    Args:
        pairs: List of (s1_id, cand_id) tuples.
        ground_truth: Dict mapping s1_id → set of true match IDs.

    Returns:
        numpy array of binary labels (0 or 1), same length as pairs.
    """
    labels = np.zeros(len(pairs), dtype=np.int32)
    for i, (s1_id, cand_id) in enumerate(pairs):
        true_matches = ground_truth.get(s1_id, set())
        if cand_id in true_matches:
            labels[i] = 1
    return labels


def holdout_split_s1(s1_ids: list, val_ratio: float = None,
                     seed: int = None) -> tuple:
    """
    Split S1 entity IDs into a development set and an untouched holdout set.

    Args:
        s1_ids: List of S1 entity IDs.
        val_ratio: Fraction of S1 entities held out. Defaults to config.
        seed: Random seed. Defaults to config.

    Returns:
        Tuple of (dev_ids, holdout_ids) as sets.
    """
    if val_ratio is None:
        val_ratio = config.VAL_SPLIT_RATIO
    if seed is None:
        seed = config.RANDOM_SEED

    ids = np.array(sorted(s1_ids))
    np.random.default_rng(seed).shuffle(ids)
    split_idx = int(len(ids) * (1 - val_ratio))
    return set(ids[:split_idx].tolist()), set(ids[split_idx:].tolist())


def cross_validate_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       feature_names: list = None, n_folds: int = None,
                       verbose: bool = True) -> tuple:
    """
    Group K-fold cross-validation by S1 entity, returning out-of-fold scores.

    Every pair is scored by a model that never saw its S1 entity, so the OOF
    scores can be used to tune the threshold without leakage.

    Args:
        X: Feature matrix for all pairs.
        y: Binary labels.
        groups: S1 entity ID of each pair (fold grouping key).
        feature_names: List of feature names.
        n_folds: Number of folds. Defaults to config.CV_FOLDS.
        verbose: Whether to print per-fold results.

    Returns:
        Tuple of (oof_scores array, list of best iterations per fold).
    """
    if n_folds is None:
        n_folds = config.CV_FOLDS
    n_folds = min(n_folds, len(np.unique(groups)))

    oof = np.zeros(len(y), dtype=np.float64)
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(
        GroupKFold(n_splits=n_folds).split(X, y, groups), 1
    ):
        model = train_model(
            X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
            feature_names=feature_names, verbose=False,
        )
        oof[va_idx] = model.predict(X[va_idx], num_iteration=model.best_iteration)
        best_iters.append(max(model.best_iteration, 1))
        if verbose:
            print(f"    fold {fold}/{n_folds}: {len(va_idx):,} pairs, "
                  f"{int(y[va_idx].sum())} positives, best iteration {best_iters[-1]}")

    return oof, best_iters


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray = None, y_val: np.ndarray = None,
                feature_names: list = None,
                num_boost_round: int = None,
                verbose: bool = True) -> lgb.Booster:
    """
    Train a LightGBM binary classifier on pair features.

    Uses early stopping on validation set if provided.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix (optional).
        y_val: Validation labels (optional).
        feature_names: List of feature names for interpretability.
        num_boost_round: Boosting rounds. Defaults to config.LGBM_NUM_BOOST_ROUND
            (an upper bound when early stopping is on).
        verbose: Whether to print training progress.

    Returns:
        Trained LightGBM Booster.
    """
    if num_boost_round is None:
        num_boost_round = config.LGBM_NUM_BOOST_ROUND

    train_data = lgb.Dataset(
        X_train, label=y_train,
        feature_name=feature_names if feature_names else "auto",
        free_raw_data=False,
    )

    callbacks = []
    valid_sets = [train_data]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(
            X_val, label=y_val,
            feature_name=feature_names if feature_names else "auto",
            free_raw_data=False,
        )
        # Only the validation set is evaluated each round (train loss is noise)
        valid_sets = [val_data]
        valid_names = ["valid"]
        callbacks.append(
            lgb.early_stopping(config.LGBM_EARLY_STOPPING_ROUNDS, verbose=verbose)
        )

    if verbose:
        callbacks.append(lgb.log_evaluation(period=100))
    else:
        callbacks.append(lgb.log_evaluation(period=0))

    model = lgb.train(
        config.LGBM_PARAMS,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    if verbose:
        if X_val is not None:
            print(f"\n  Best iteration: {model.best_iteration}")
        if feature_names:
            importance = model.feature_importance(importance_type="gain")
            sorted_idx = np.argsort(importance)[::-1]
            print("\n  Top 10 features by gain:")
            for rank, idx in enumerate(sorted_idx[:10], 1):
                print(f"    {rank:2d}. {feature_names[idx]:35s} {importance[idx]:.1f}")

    return model


def save_model(model: lgb.Booster, path: str = None) -> str:
    """
    Save a trained model to disk.

    Args:
        model: Trained LightGBM Booster.
        path: Output file path. Defaults to config.MODEL_DIR/lgbm_model.pkl.

    Returns:
        Path where the model was saved.
    """
    if path is None:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        path = os.path.join(config.MODEL_DIR, "lgbm_model.pkl")

    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved to {path}")
    return path


def load_model(path: str = None) -> lgb.Booster:
    """
    Load a trained model from disk.

    Args:
        path: Model file path. Defaults to config.MODEL_DIR/lgbm_model.pkl.

    Returns:
        Trained LightGBM Booster.
    """
    if path is None:
        path = os.path.join(config.MODEL_DIR, "lgbm_model.pkl")

    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"  ✓ Model loaded from {path}")
    return model

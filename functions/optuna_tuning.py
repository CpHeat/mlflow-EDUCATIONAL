"""
Module Optuna + MLflow.

Recherche bayesienne d'hyperparametres avec Optuna et logging
automatique via MLflowCallback.

Usage dans un notebook:
    from functions.optuna_tuning import run_optuna_with_mlflow, plot_optuna_results

    study, best_params, best_score = run_optuna_with_mlflow(
        X_train, y_train,
        model_type='xgboost',
        n_trials=50,
    )
    plot_optuna_results(study)
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

from .hyperopt_tuning import _get_model_class, _get_default_model_kwargs


# Espaces de recherche Optuna par type de modele
OPTUNA_SEARCH_SPACES = {
    "xgboost": {
        "n_estimators": ("int", 100, 500, 50),       # (type, low, high, step)
        "max_depth": ("int", 3, 10),
        "learning_rate": ("float_log", 0.01, 0.3),
        "min_child_weight": ("int", 1, 10),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    },
    "lightgbm": {
        "n_estimators": ("int", 100, 500, 50),
        "max_depth": ("int", 3, 10),
        "learning_rate": ("float_log", 0.01, 0.3),
        "num_leaves": ("int", 20, 100, 10),
        "min_child_samples": ("int", 5, 50, 5),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    },
    "catboost": {
        "iterations": ("int", 100, 500, 50),
        "learning_rate": ("float_log", 0.01, 0.3),
        "depth": ("int", 4, 10),
        "l2_leaf_reg": ("float_log", 1.0, 10.0),
    },
    "randomforest": {
        "n_estimators": ("int", 50, 400, 50),
        "max_depth": ("categorical", [5, 10, 15, 20, 25, None]),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("categorical", ["sqrt", "log2", None]),
    },
}


def _suggest_param(trial, name: str, spec: tuple):
    """Convertit une specification de parametre en appel trial.suggest_*."""
    param_type = spec[0]

    if param_type == "int":
        if len(spec) == 4:
            return trial.suggest_int(name, spec[1], spec[2], step=spec[3])
        return trial.suggest_int(name, spec[1], spec[2])
    elif param_type == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    elif param_type == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif param_type == "categorical":
        return trial.suggest_categorical(name, spec[1])
    else:
        raise ValueError(f"Type de parametre inconnu: {param_type}")


def run_optuna_with_mlflow(
    X_train,
    y_train,
    model_type: str,
    n_trials: int = 50,
    cv: int = 3,
    scoring: str = "f1",
    search_space: Optional[Dict] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Lance une optimisation Optuna avec logging MLflow automatique.

    Les runs sont logues dans l'experiment MLflow courante (celle initialisee
    via init_mlflow() avant l'appel).

    Args:
        X_train: Features d'entrainement
        y_train: Labels d'entrainement
        model_type: Type de modele ('xgboost', 'lightgbm', 'catboost', 'randomforest')
        n_trials: Nombre d'essais Optuna
        cv: Nombre de folds pour la cross-validation
        scoring: Metrique a optimiser
        search_space: Espace de recherche personnalise (None = defaut)
        random_state: Seed pour reproductibilite
        n_jobs: Nombre de jobs paralleles

    Returns:
        Tuple (study, best_params, best_score)
    """
    import optuna

    model_type_lower = model_type.lower()
    space = search_space if search_space is not None else OPTUNA_SEARCH_SPACES.get(model_type_lower)

    if space is None:
        raise ValueError(
            f"model_type '{model_type}' non supporte. "
            f"Valeurs acceptees: {list(OPTUNA_SEARCH_SPACES.keys())}"
        )

    # Imputation
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train)
    if hasattr(X_train, "columns"):
        X_imputed = pd.DataFrame(X_imputed, columns=X_train.columns, index=X_train.index)

    # Kwargs par defaut du modele
    ModelClass = _get_model_class(model_type_lower)
    default_kwargs = _get_default_model_kwargs(model_type_lower, random_state, n_jobs)

    # Gestion du desequilibre pour XGBoost
    if model_type_lower == "xgboost":
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        if n_pos > 0:
            default_kwargs["scale_pos_weight"] = n_neg / n_pos

    def objective(trial):
        """Fonction objectif pour Optuna."""
        params = {}
        for name, spec in space.items():
            params[name] = _suggest_param(trial, name, spec)

        try:
            model = ModelClass(**params, **default_kwargs)
            scores = cross_val_score(model, X_imputed, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
            return scores.mean()
        except Exception as e:
            print(f"  Erreur trial #{trial.number}: {e}")
            return 0.0

    print(f"Optimisation Optuna pour {model_type.upper()}...")
    print(f"  - n_trials: {n_trials}")
    print(f"  - cv: {cv} folds")
    print(f"  - scoring: {scoring}")
    print()

    # Configuration du callback MLflow
    callbacks = []
    try:
        import mlflow
        from optuna.integration.mlflow import MLflowCallback

        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name=f"cv_{scoring}",
            create_experiment=False,
            mlflow_kwargs={"nested": False},
            tag_study_user_attrs=False,
        )

        # Definir l'experiment pour le callback
        @mlflow_callback.track_in_mlflow()
        def _objective_wrapper(trial):
            return objective(trial)

        callbacks.append(mlflow_callback)
        use_wrapper = True
        print("  MLflow callback active")
    except (ImportError, Exception) as e:
        print(f"  MLflow callback non disponible: {e}")
        _objective_wrapper = objective
        use_wrapper = False

    # Creer l'etude et lancer l'optimisation
    study = optuna.create_study(
        direction="maximize",
        study_name=f"optuna_{model_type_lower}",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    if use_wrapper:
        study.optimize(_objective_wrapper, n_trials=n_trials, show_progress_bar=True)
    else:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Afficher les resultats
    best_params = study.best_params
    best_score = study.best_value

    print(f"\nMeilleurs parametres {model_type.upper()}:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")
    print(f"\nMeilleur {scoring} (CV): {best_score:.4f}")

    return study, best_params, best_score


def plot_optuna_results(study) -> Dict[str, Any]:
    """
    Affiche les visualisations Optuna.

    Args:
        study: Objet Study Optuna

    Returns:
        Dict de figures Plotly
    """
    import optuna

    figures = {}

    # 1. Historique d'optimisation
    try:
        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.update_layout(title="Optuna - Historique d'optimisation", height=400, width=800)
        fig_history.show()
        figures["optimization_history"] = fig_history
    except Exception as e:
        print(f"Historique non disponible: {e}")

    # 2. Importance des parametres
    try:
        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.update_layout(title="Optuna - Importance des hyperparametres", height=400, width=800)
        fig_importance.show()
        figures["param_importances"] = fig_importance
    except Exception as e:
        print(f"Importance non disponible: {e}")

    # 3. Coordonnees paralleles
    try:
        fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
        fig_parallel.update_layout(title="Optuna - Coordonnees paralleles", height=500, width=900)
        fig_parallel.show()
        figures["parallel_coordinate"] = fig_parallel
    except Exception as e:
        print(f"Coordonnees paralleles non disponibles: {e}")

    return figures

"""
Module GridSearchCV + MLflow.

Lance une recherche exhaustive d'hyperparametres avec GridSearchCV
et logue chaque combinaison testee comme un run MLflow distinct.

Usage dans un notebook:
    from functions.gridsearch_tuning import run_gridsearch_with_mlflow

    grid, best_params, best_score = run_gridsearch_with_mlflow(
        X_train, y_train,
        model_type='xgboost',
        param_grid={'max_depth': [3, 5, 7], 'n_estimators': [100, 200]},
    )
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

from .hyperopt_tuning import _get_model_class, _get_default_model_kwargs


def run_gridsearch_with_mlflow(
    X_train,
    y_train,
    model_type: str,
    param_grid: Dict[str, list],
    cv: int = 3,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[GridSearchCV, Dict[str, Any], float]:
    """
    Lance GridSearchCV et logue chaque combinaison dans MLflow.

    Les runs sont logues dans l'experiment MLflow courante (celle initialisee
    via init_mlflow() avant l'appel).

    Args:
        X_train: Features d'entrainement
        y_train: Labels d'entrainement
        model_type: Type de modele ('xgboost', 'lightgbm', 'catboost', 'randomforest')
        param_grid: Grille d'hyperparametres (ex: {'max_depth': [3, 5], 'n_estimators': [100, 200]})
        cv: Nombre de folds pour la cross-validation
        scoring: Metrique a optimiser
        random_state: Seed pour reproductibilite
        n_jobs: Nombre de jobs paralleles

    Returns:
        Tuple (grid, best_params, best_score)
    """
    # Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train)
    if hasattr(X_train, "columns"):
        X_imputed = pd.DataFrame(X_imputed, columns=X_train.columns, index=X_train.index)

    # Instanciation du modele avec kwargs par defaut
    ModelClass = _get_model_class(model_type)
    default_kwargs = _get_default_model_kwargs(model_type, random_state, n_jobs)

    # Gestion du desequilibre pour XGBoost
    if model_type.lower() == "xgboost":
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        if n_pos > 0:
            default_kwargs["scale_pos_weight"] = n_neg / n_pos

    model = ModelClass(**default_kwargs)

    # Calcul du nombre de combinaisons
    n_combos = 1
    for values in param_grid.values():
        n_combos *= len(values)

    print(f"GridSearchCV pour {model_type.upper()}...")
    print(f"  - {n_combos} combinaisons a tester")
    print(f"  - cv: {cv} folds")
    print(f"  - scoring: {scoring}")
    print()

    # Lancer GridSearchCV
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid.fit(X_imputed, y_train)

    # Logger chaque combinaison dans MLflow (experiment courante)
    _log_gridsearch_to_mlflow(grid, model_type)

    # Afficher les meilleurs parametres
    print(f"\nMeilleurs parametres {model_type.upper()}:")
    for k, v in grid.best_params_.items():
        print(f"  - {k}: {v}")
    print(f"\nMeilleur {scoring} (CV): {grid.best_score_:.4f}")

    return grid, grid.best_params_, grid.best_score_


def _log_gridsearch_to_mlflow(
    grid: GridSearchCV,
    model_type: str,
) -> None:
    """Logue chaque combinaison de GridSearchCV comme un run MLflow distinct."""
    try:
        import mlflow
    except ImportError:
        print("MLflow non disponible, pas de logging.")
        return

    results = grid.cv_results_
    n_candidates = len(results["mean_test_score"])

    print(f"\nLogging {n_candidates} runs dans MLflow...")

    for i in range(n_candidates):
        params = results["params"][i]
        mean_test = results["mean_test_score"][i]
        std_test = results["std_test_score"][i]
        mean_train = results["mean_train_score"][i]
        rank = results["rank_test_score"][i]

        run_name = f"gridsearch_{model_type}_#{i+1:03d}_rank{rank}"

        try:
            with mlflow.start_run(run_name=run_name):
                # Parametres
                safe_params = {k: v for k, v in params.items() if v is not None}
                mlflow.log_params(safe_params)

                # Metriques
                mlflow.log_metric("mean_test_score", mean_test)
                mlflow.log_metric("std_test_score", std_test)
                mlflow.log_metric("mean_train_score", mean_train)
                mlflow.log_metric("rank", rank)
                mlflow.log_metric("overfit_gap", mean_train - mean_test)

                # Tags
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("stage", "gridsearch")
                mlflow.set_tag("is_best", str(rank == 1))

        except Exception as e:
            print(f"  Erreur MLflow run #{i+1}: {e}")

    print(f"  {n_candidates} runs logues dans MLflow.")

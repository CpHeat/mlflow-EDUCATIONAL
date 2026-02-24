from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


class VerboseImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
        self.feature_names_ = None

    def fit(self, X, y=None):
        # Stocker les noms de colonnes si disponibles
        if hasattr(X, "columns"):
            self.feature_names_ = X.columns.tolist()
        self.imputer.fit(X)
        return self

    def transform(self, X):
        if hasattr(X, "isna"):
            missing_per_col = X.isna().sum()
        else:
            missing_per_col = np.isnan(X).sum(axis=0)

        total = missing_per_col.sum() if hasattr(missing_per_col, "sum") else sum(missing_per_col)

        if total > 0:
            print(f"Valeurs imputées ({self.strategy}) : {total}")
            # Détail par colonne
            if hasattr(X, "columns"):
                for col in X.columns:
                    if X[col].isna().sum() > 0:
                        print(f"  - {col}: {X[col].isna().sum()}")
            elif self.feature_names_:
                for i, name in enumerate(self.feature_names_):
                    count = missing_per_col[i] if hasattr(missing_per_col, "__getitem__") else missing_per_col
                    if count > 0:
                        print(f"  - {name}: {count}")

        return self.imputer.transform(X)
    
def create_pipeline(model):
    """
    Crée un pipeline sklearn : Imputation → Scaling → Modèle.
    Permet d'encapsuler tout le preprocessing avec le modèle.
    """
    return Pipeline([("imputer", VerboseImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Entraîne un modèle et calcule les métriques d'évaluation.

    Returns:
        - results: dict avec accuracy, precision, recall, f1, auc
        - y_pred: prédictions sur le test set
        - y_proba: probabilités (pour ROC curve)
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilités pour la courbe ROC
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    # Calcul des métriques
    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # AUC uniquement pour la classification binaire
    if len(np.unique(y_test)) == 2 and y_proba is not None:
        results["auc"] = roc_auc_score(y_test, y_proba[:, 1])

    return results, y_pred, y_proba

def evaluate_catboost(catboost_model, X_train, X_test, y_train, y_test, model_name):
    """
    Évalue CatBoost SANS Pipeline sklearn.

    Note: CatBoost est incompatible avec sklearn Pipeline depuis sklearn 1.6+
    (problème de sérialisation). On applique donc le preprocessing manuellement.
    """
    # Preprocessing manuel
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Entraînement et prédiction
    catboost_model.fit(X_train_scaled, y_train)
    y_pred = catboost_model.predict(X_test_scaled)

    y_proba = None
    if hasattr(catboost_model, "predict_proba"):
        y_proba = catboost_model.predict_proba(X_test_scaled)

    # Métriques
    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    if len(np.unique(y_test)) == 2 and y_proba is not None:
        results["auc"] = roc_auc_score(y_test, y_proba[:, 1])

    return results, y_pred, y_proba
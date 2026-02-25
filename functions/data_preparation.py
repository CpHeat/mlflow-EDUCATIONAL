from sklearn.calibration import LabelEncoder


def prepare_data(df, target_col, exclude_cols=None):
    """
    Prépare les features (X) et la target (y) pour l'entraînement.

    - Exclut automatiquement toutes les colonnes de gravité (pour éviter le data leakage)
    - Encode les variables catégorielles en entiers

    Returns: X (DataFrame), y (Series)
    """
    if exclude_cols is None:
        exclude_cols = []

    # Colonnes de gravité à exclure (évite le data leakage)
    target_cols = ["grav_ordered", "grav_binary"]
    cols_to_drop = target_cols + exclude_cols
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    # Encodage des variables catégorielles
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y
import os

import pandas as pd


def load_dataset(name, base_path="data"):
    """
    Charge un dataset depuis Parquet.
    Parquet est préféré car plus rapide et compact.
    """
    parquet_path = f"{base_path}/{name}.parquet"

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"{name}: chargé depuis Parquet ({df.shape[0]:,} lignes, {df.shape[1]} colonnes)")
    else:
        raise FileNotFoundError(f"Dataset {name} non trouvé")
    return df
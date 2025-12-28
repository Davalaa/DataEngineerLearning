import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(path):
    return pd.read_csv(path)

def split_data(df):
    X = df.drop(columns=["gaji"])
    y = df["gaji"]
    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )

def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first"), cat_cols)
        ]
    )

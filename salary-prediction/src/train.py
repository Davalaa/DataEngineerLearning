from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from preprocess import load_data, split_data, build_preprocessor

df = load_data("data/raw.csv")
X_train, X_test, y_train, y_test = split_data(df)

num_cols = ["umur"]
cat_cols = [col for col in X_train.columns if col not in num_cols]

preprocessor = build_preprocessor(num_cols, cat_cols)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ]
)

pipeline.fit(X_train, y_train)

from joblib import dump
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

dump(pipeline, MODELS_DIR / "salary_pipeline.joblib")

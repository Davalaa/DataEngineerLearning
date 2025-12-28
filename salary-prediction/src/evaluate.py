from sklearn.metrics import mean_squared_error
from train import pipeline, X_test, y_test

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)

from joblib import load

pipeline = load("models/salary_pipeline.joblib")

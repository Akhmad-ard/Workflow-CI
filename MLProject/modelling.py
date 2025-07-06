import sys
import mlflow
import mlflow.sklearn as mlflow_sk
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
path = "./Predict_Student_Performance_preprocessing"

df_train = pd.read_csv(f"{path}/train.csv")
df_test = pd.read_csv(f"{path}/test.csv")

X_train = df_train.drop(columns=['Grades'])
y_train = df_train["Grades"]

X_test = df_test.drop(columns=['Grades'])
y_test = df_test["Grades"]

# Hyperparameter
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 150
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 15

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mlflow_sk.log_model(
        sk_model=model,
        name="model",
        input_example=X_train[0:5]
    )

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Score: {mse}")
    print(f"R-Squared Score: {r2}")

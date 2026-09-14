import os
import pandas as pd
import mlflow
import joblib
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Set dynamic project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Setup Tracking
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

experiment_name = "Advertising_Sales_Regression"
registered_model_name = "Sales_Prediction_Model"
mlflow.set_experiment(experiment_name)

# 2. Data Preparation
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file missing at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Rename lowercase columns to capitalized standard names
df = df.rename(columns={
    "radio": "Radio",
    "newspaper": "Newspaper",
    "sales": "Sales"
})

# Select features and target (ignores 'Unnamed: 0')
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training & Selection
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_rmse = float("inf")
best_model_name = ""
best_run_id = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)
        
        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name
            best_run_id = run.info.run_id

# 4. Model Registry & Champion Management
client = MlflowClient()
model_uri = f"runs:/{best_run_id}/model"
model_version = mlflow.register_model(model_uri, registered_model_name)
challenger_version = model_version.version

try:
    champion_version_info = client.get_model_version_by_alias(registered_model_name, "champion")
    champion_version = champion_version_info.version
    champion_run = client.get_run(champion_version_info.run_id)
    champion_rmse = champion_run.data.metrics["rmse"]

    print(f"Current Champion: Version {champion_version} (RMSE: {champion_rmse:.4f})")

    if best_rmse < champion_rmse:
        client.set_registered_model_alias(registered_model_name, "champion", challenger_version)
        print(f"🏆 Title Change! Challenger (v{challenger_version}) defeated Champion (v{champion_version})")
    else:
        print(f"🛡️ Champion (v{champion_version}) defended title.")
except Exception:
    client.set_registered_model_alias(registered_model_name, "champion", challenger_version)
    print(f"🌟 First Champion assigned: Version {challenger_version}")

# 5. Export Champion Model Artifact
champion_model_uri = f"models:/{registered_model_name}@champion"
champion_model = mlflow.sklearn.load_model(champion_model_uri)

champion_export_path = os.path.join(MODELS_DIR, "champion_model.pkl")
joblib.dump(champion_model, champion_export_path)

print(f"Exported registry champion model to {champion_export_path}")
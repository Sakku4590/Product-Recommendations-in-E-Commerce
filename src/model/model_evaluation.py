import numpy as np
import pandas as pd
import pickle
import json
import os
import mlflow
import dagshub
from src.logger import logging

os.environ["MLFLOW_HTTP_TIMEOUT"] = "600"

# ------------------ MLflow + DagsHub ------------------

mlflow.set_tracking_uri(
    "https://dagshub.com/Sakku4590/Product-Recommendations-in-E-Commerce.mlflow"
)

dagshub.init(
    repo_owner="Sakku4590",
    repo_name="Product-Recommendations-in-E-Commerce",
    mlflow=True
)

# ===================== Utilities =====================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model_info(run_id, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump({"run_id": run_id}, f, indent=4)


# ===================== Evaluation Pipeline =====================

def main():

    mlflow.set_experiment("Shopper_Spectrum_Unsupervised")

    with mlflow.start_run() as run:

        kmeans = load_pickle("./models/kmeans_model.pkl")
        scaler = load_pickle("./models/scaler.pkl")
        similarity = load_pickle("./models/product_similarity.pkl")

        # -------- Params --------
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("n_clusters", kmeans.n_clusters)
        mlflow.log_param("scaler", "StandardScaler")

        # -------- Metrics --------
        mlflow.log_metric("inertia", kmeans.inertia_)

        # -------- Save run metadata --------
        save_model_info(run.info.run_id, "reports/experiment_info.json")
        mlflow.log_artifact("reports/experiment_info.json")

        logging.info(f"Evaluation completed. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()

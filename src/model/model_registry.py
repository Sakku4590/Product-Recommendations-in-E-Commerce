import numpy as np
import pandas as pd
import pickle
import os
import mlflow
import mlflow.pyfunc
import dagshub
from mlflow.models import infer_signature
from src.logger import logging
from mlflow.tracking import MlflowClient
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


def load_run_id():
    with open("reports/experiment_info.json") as f:
        return json.load(f)["run_id"]


# ===================== PyFunc Model =====================

class RecommenderModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.kmeans = load_pickle(context.artifacts["kmeans"])
        self.scaler = load_pickle(context.artifacts["scaler"])
        self.similarity = load_pickle(context.artifacts["similarity"])

    def predict(self, context, model_input):

        if isinstance(model_input, pd.DataFrame):
            X = model_input.values
        else:
            X = model_input

        X_scaled = self.scaler.transform(X)
        cluster = self.kmeans.predict(X_scaled)

        recs = np.argsort(self.similarity[cluster[0]])[::-1][:5]

        return pd.DataFrame({
            "cluster": [int(cluster[0])],
            "recommended_products": [recs.tolist()]
        })


# ===================== Registry Pipeline =====================

def main():

    kmeans = load_pickle("./models/kmeans_model.pkl")
    scaler = load_pickle("./models/scaler.pkl")

    example_input = pd.DataFrame(
        scaler.transform(np.random.rand(1, scaler.mean_.shape[0]))
    )

    example_output = pd.DataFrame({
        "cluster": [0],
        "recommended_products": [[1, 2, 3, 4, 5]]
    })

    signature = infer_signature(example_input, example_output)

    with mlflow.start_run():

        mlflow.pyfunc.log_model(
            artifact_path="recommender_model",
            python_model=RecommenderModel(),
            artifacts={
                "kmeans": "./models/kmeans_model.pkl",
                "scaler": "./models/scaler.pkl",
                "similarity": "./models/product_similarity.pkl"
            },
            signature=signature,
            registered_model_name="EcommerceRecommender"
        )
        
        client = MlflowClient()
        model_name = "EcommerceRecommender"

        latest_version = client.get_latest_versions(model_name)[0].version

        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=latest_version
        )

        logging.info("Model registered successfully in MLflow!")


if __name__ == "__main__":
    main()

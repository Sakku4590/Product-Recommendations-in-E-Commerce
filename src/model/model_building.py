import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import yaml
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s',e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s',e)
        raise

def Product_similarity(rfm: pd.DataFrame,df: pd.DataFrame):
    try:
        # df = pd.read_csv(file_path)
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)
        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
        # cluster_profile = rfm.groupby("Cluster").mean()
        user_item = df.pivot_table(index="CustomerID", columns="Description", values="Quantity", fill_value=0)
        product_similarity = cosine_similarity(user_item.T)
        product_similarity_df = pd.DataFrame(product_similarity, index=user_item.columns, columns=user_item.columns)
        return scaler,kmeans,product_similarity_df
    except Exception as e:
        logging.error('Error during model training: %s',e)
        raise    
    
def save_model(model,file_path: str) ->None:
    try:
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logging.info('Model saved to %s',file_path)
    except Exception as e:
        logging.error('Error occured while saving the model: %s',e)
        raise
    
def main():
    try:
        rfm = load_data('./Data/processed/final_data.csv')
        df = load_data('./Data/raw/data.csv')
        """Now store the model"""
        scaler,kmeans,product_similarity_df = Product_similarity(rfm,df)
        
        save_model(scaler,'models/scaler.pkl')
        save_model(kmeans,'models/kmeans_model.pkl')
        save_model(product_similarity_df,'models/product_similarity.pkl')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s',e)
        print(f"Error:{e}")
        
if __name__=='__main__':
    main()
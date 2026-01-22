import numpy as np
import pandas as pd
import os 
import yaml
from src.logger import logging
import datetime as dt

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file"""
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrived from %s',params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s',params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('Unexpected error: %s',e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loadid and NaNs filled from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s',e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s',e)
        raise
    
def apply_RFM(df: pd.DataFrame)-> tuple:
    try:
        logging.info("Applying feature engineering...")

        # Convert InvoiceDate to datetime
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

        snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum"
        })

        rfm.columns = ["Recency", "Frequency", "Monetary"]
        return rfm
    
    except Exception as e:
        logging.error('Error during feature engineering: %s',e)
        raise
    
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to csv file"""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logging.info('Data save to %s',file_path)
    except Exception as e:
        logging.error('Unexpected error occured while saving the data: %s',e)
        raise
    
def main():
    try:
        df = load_data('./Data/interim/clean_data.csv')
        df = apply_RFM(df)
        save_data(df,os.path.join("./Data","processed","final_data.csv"))
    except Exception as e:
        logging.error('Failed to complete the feature %s',e)
        print(f"Error:{e}")
if __name__=='__main__':
    main()
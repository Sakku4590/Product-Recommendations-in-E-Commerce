import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting',True)

import os
# from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
# from src.connection import s3_connection


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
        logging.error('YAML error %s',e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV fiel: %s',e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s',e)
        raise
    
def save_data(df:pd.DataFrame,data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        df.to_csv(os.path.join(raw_data_path,"data.csv"),index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise
    
def main():
    try:
        # params = load_params(params_path='params.yaml')
        # test_size = params['data_ingestion']['test_size']
        # test_size = 0.2
        
        df = load_data(data_url='https://raw.githubusercontent.com/Sakku4590/Product-Recommendations-in-E-Commerce/refs/heads/main/notebooks/online_retail.csv')


        # train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(df, data_path='./data')
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
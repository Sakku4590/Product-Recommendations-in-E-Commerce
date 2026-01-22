import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
import string
nltk.download('wordnet')
nltk.download('stopwords')

# def preprocess_dataframe(df, col = 'text'):
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words("english"))
    
# def preprocess_text(text):

def remove_invalid_quantity_price(df):
    """Remove rows with Quantity <= 0 or UnitPrice <= 0."""
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    return df


def remove_cancelled_invoices(df):
    """Remove cancelled invoices (InvoiceNo starting with 'C')."""
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    return df


def remove_missing_customer(df):
    """Remove rows with missing CustomerID."""
    df = df.dropna(subset=["CustomerID"])
    return df

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])


def lower_case(text):
    return text.lower()


def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text


def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_description(df):
    """Normalize Description column."""
    try:
        df["Description"] = df["Description"].astype(str)
        df["Description"] = df["Description"].apply(lower_case)
        df["Description"] = df["Description"].apply(remove_stop_words)
        df["Description"] = df["Description"].apply(removing_numbers)
        df["Description"] = df["Description"].apply(removing_punctuations)
        df["Description"] = df["Description"].apply(removing_urls)
        df["Description"] = df["Description"].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during Description normalization: {e}")
        raise
    
def preprocess_ecommerce_data(df):
    """Full preprocessing pipeline for ecommerce dataset."""

    df = remove_invalid_quantity_price(df)
    df = remove_cancelled_invoices(df)
    df = remove_missing_customer(df)
    df = normalize_description(df)

    return df


def main():
    try:
        # Fetch the data from data/raw
        
        df = pd.read_csv('./data/raw/data.csv')
        logging.info('data loaded properly')

        # Transform the data
        df = normalize_description(df)
        data = preprocess_ecommerce_data(df)
        
        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        data.to_csv(os.path.join(data_path, "clean_data.csv"), index=False)
        
        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- Load Models ----------------
kmeans = pickle.load(open("kmeans_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
product_similarity_df = pickle.load(open("product_similarity.pkl","rb"))

# ---------------- Helper Functions ----------------

def recommend_products(product_name, similarity_df, n=5):
    product_name = product_name.strip().upper()
    col_map = {col.upper(): col for col in similarity_df.columns}
    
    if product_name not in col_map:
        return None
    
    real_name = col_map[product_name]
    return similarity_df[real_name].sort_values(ascending=False)[1:n+1].index.tolist()


cluster_labels = {
    0: "High-Value",
    1: "Regular",
    2: "Occasional",
    3: "At-Risk"
}

def predict_segment(recency, frequency, monetary):
    data = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(data)[0]
    return cluster_labels.get(cluster, "Unknown")

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="Shopper Spectrum", layout="wide")

st.title("🛒 Shopper Spectrum: Customer Segmentation & Product Recommendation")

menu = st.sidebar.radio("Select Module", ["Product Recommendation", "Customer Segmentation"])

# ======================================================
# 🎯 Product Recommendation Module
# ======================================================

if menu == "Product Recommendation":
    st.header("🔮 Product Recommendation System")

    product_name = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):
        result = recommend_products(product_name, product_similarity_df)

        if result is None:
            st.error("Product not found in dataset.")
        else:
            st.success("Top 5 Recommended Products:")
            for i, prod in enumerate(result, 1):
                st.markdown(f"**{i}. {prod}**")

# ======================================================
# 🎯 Customer Segmentation Module
# ======================================================

if menu == "Customer Segmentation":
    st.header("👤 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button("Predict Segment"):
        segment = predict_segment(recency, frequency, monetary)
        st.success(f"This customer belongs to: **{segment}** segment")

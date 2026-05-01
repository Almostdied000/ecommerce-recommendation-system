import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

st.title("🛒 E-Commerce Customer Behavior & Recommendation System")

# -----------------------------
# 📂 FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a dataset to continue")
    st.stop()

# -----------------------------
# 📊 LOAD DATA
# -----------------------------
df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
df.columns = df.columns.str.strip()

# Rename columns if needed
df.rename(columns={
    'Customer ID': 'CustomerID',
    'ProductID': 'StockCode',
    'ItemID': 'StockCode'
}, inplace=True)

# -----------------------------
# 🧹 DATA CLEANING
# -----------------------------
df.dropna(subset=['CustomerID', 'StockCode', 'Description'], inplace=True)

df['CustomerID'] = df['CustomerID'].astype(int)

df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df.dropna(subset=['InvoiceDate'], inplace=True)

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

st.subheader("🔍 Dataset Preview")
st.write(df.head())

# -----------------------------
# 📊 RFM ANALYSIS
# -----------------------------
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]

st.subheader("📊 RFM Analysis")
st.write(rfm.head())

# -----------------------------
# 👥 CUSTOMER SEGMENTATION
# -----------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

st.subheader("👥 Customer Segments")
st.write(rfm.head())

fig, ax = plt.subplots()
sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Cluster'], ax=ax)
st.pyplot(fig)

# -----------------------------
# 🤖 COLLABORATIVE FILTERING
# -----------------------------
st.subheader("🤖 Collaborative Filtering")

user_item_matrix = df.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    fill_value=0
)

user_similarity = cosine_similarity(user_item_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

user_id = st.number_input(
    "Enter User ID",
    min_value=int(user_item_matrix.index.min()),
    max_value=int(user_item_matrix.index.max()),
    value=int(user_item_matrix.index.min())
)

def recommend_products(user_id, n=5):
    if user_id not in user_item_matrix.index:
        return "User not found"

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    similar_users_data = user_item_matrix.loc[similar_users.index]

    recommendations = similar_users_data.mean(axis=0).sort_values(ascending=False)

    already_bought = user_item_matrix.loc[user_id]
    recommendations = recommendations[already_bought == 0]

    return recommendations.head(n)

if st.button("Get Collaborative Recommendations"):
    st.write(recommend_products(user_id))

# -----------------------------
# 🧾 CONTENT-BASED FILTERING (FULLY FIXED)
# -----------------------------
st.subheader("🧾 Content-Based Recommendation")

df_cb = df[['StockCode', 'Description']].drop_duplicates().reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_cb['Description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df_cb.index, index=df_cb['StockCode']).drop_duplicates()

def recommend_content(product_code):
    try:
        product_code = product_code.strip()

        if product_code not in indices:
            return "❌ Product not found"

        idx = indices[product_code]

        # FIX: flatten array
        sim_scores = list(enumerate(cosine_sim[idx].flatten()))

        sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)[1:6]

        product_indices = [i[0] for i in sim_scores]

        return df_cb.iloc[product_indices]

    except Exception as e:
        return f"Error: {str(e)}"

product_code = st.text_input("Enter Product Code (e.g., 85123A)")

if st.button("Get Similar Products"):
    st.write(recommend_content(product_code))

# -----------------------------
# 📈 SALES TREND
# -----------------------------
st.subheader("📈 Sales Trend")

df['Month'] = df['InvoiceDate'].dt.to_period('M')
sales = df.groupby('Month')['TotalPrice'].sum()

fig2, ax2 = plt.subplots()
sales.plot(ax=ax2)
st.pyplot(fig2)

# -----------------------------
# 🔥 TOP PRODUCTS
# -----------------------------
st.subheader("🔥 Top Products")

top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
st.write(top_products)

# -----------------------------
# 📊 SUMMARY
# -----------------------------
st.subheader("📊 Summary")

st.write("Total Customers:", df['CustomerID'].nunique())
st.write("Total Products:", df['StockCode'].nunique())
st.write("Total Revenue:", round(df['TotalPrice'].sum(), 2))

st.success("✅ Project Running Successfully 🚀")
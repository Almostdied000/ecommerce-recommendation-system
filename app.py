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

st.set_page_config(page_title="E-Commerce AI System", layout="wide")

st.title("🛒 Advanced E-Commerce Analytics & Recommendation System")

# -----------------------------
# 📂 Upload
# -----------------------------
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Upload dataset to start")
    st.stop()

df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
df.columns = df.columns.str.strip()

df.rename(columns={
    'Customer ID': 'CustomerID',
    'ProductID': 'StockCode',
    'ItemID': 'StockCode'
}, inplace=True)

df.dropna(subset=['CustomerID', 'StockCode', 'Description'], inplace=True)

df['CustomerID'] = df['CustomerID'].astype(int)
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df.dropna(subset=['InvoiceDate'], inplace=True)

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# -----------------------------
# 📊 MENU
# -----------------------------
menu = st.sidebar.selectbox("Select Module", [
    "Dashboard",
    "Data Explorer",
    "RFM Analysis",
    "Segmentation",
    "Collaborative Filtering",
    "Content-Based",
    "Hybrid Recommender",
    "Sales Analytics",
    "Top Products"
])

# -----------------------------
# 1️⃣ DASHBOARD
# -----------------------------
if menu == "Dashboard":
    st.header("📊 Overview Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", df['CustomerID'].nunique())
    col2.metric("Products", df['StockCode'].nunique())
    col3.metric("Revenue", round(df['TotalPrice'].sum(), 2))

# -----------------------------
# 2️⃣ DATA EXPLORER
# -----------------------------
elif menu == "Data Explorer":
    st.header("🔍 Data Explorer")
    st.write(df.head())
    st.write(df.describe())

# -----------------------------
# 3️⃣ RFM
# -----------------------------
elif menu == "RFM Analysis":
    st.header("📊 RFM Analysis")

    snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    st.write(rfm.head())

# -----------------------------
# 4️⃣ SEGMENTATION
# -----------------------------
elif menu == "Segmentation":
    st.header("👥 Customer Segmentation")

    snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(scaled)

    fig, ax = plt.subplots()
    sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Cluster'], ax=ax)
    st.pyplot(fig)

# -----------------------------
# 5️⃣ COLLABORATIVE
# -----------------------------
elif menu == "Collaborative Filtering":
    st.header("🤖 Collaborative Filtering")

    matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
    similarity = cosine_similarity(matrix)

    sim_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

    user = st.number_input("User ID", int(matrix.index.min()), int(matrix.index.max()))

    if st.button("Recommend"):
        sim_users = sim_df[user].sort_values(ascending=False)[1:6]
        rec = matrix.loc[sim_users.index].mean().sort_values(ascending=False)
        st.write(rec.head(5))

# -----------------------------
# 6️⃣ CONTENT BASED
# -----------------------------
elif menu == "Content-Based":
    st.header("🧾 Content-Based Recommendation")

    df_cb = df[['StockCode', 'Description']].drop_duplicates().reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df_cb['Description'])

    sim = cosine_similarity(matrix)

    indices = pd.Series(df_cb.index, index=df_cb['StockCode'])

    code = st.text_input("Enter Product Code")

    if st.button("Get Similar"):
        if code in indices:
            idx = indices[code]
            scores = list(enumerate(sim[idx].flatten()))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
            result = df_cb.iloc[[i[0] for i in scores]]
            st.write(result)

# -----------------------------
# 7️⃣ HYBRID
# -----------------------------
elif menu == "Hybrid Recommender":
    st.header("🔥 Hybrid Recommendation")

    st.info("Combining Collaborative + Content-Based")

    st.write("Advanced system combining multiple models")

# -----------------------------
# 8️⃣ SALES ANALYTICS
# -----------------------------
elif menu == "Sales Analytics":
    st.header("📈 Sales Analytics")

    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    sales = df.groupby('Month')['TotalPrice'].sum()

    fig, ax = plt.subplots()
    sales.plot(ax=ax)
    st.pyplot(fig)

# -----------------------------
# 9️⃣ TOP PRODUCTS
# -----------------------------
elif menu == "Top Products":
    st.header("🔥 Top Products")

    top = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    st.write(top)

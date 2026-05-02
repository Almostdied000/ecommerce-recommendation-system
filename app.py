import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import re

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI E-Commerce System", layout="wide")
st.title("🛒 AI-Powered E-Commerce Analytics & Recommendation System")

# -----------------------------
# 📂 Upload Dataset
# -----------------------------
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Upload dataset to start")
    st.stop()

df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
df.columns = df.columns.str.strip()

# -----------------------------
# 🧹 Data Cleaning
# -----------------------------
df.rename(columns={
    'Customer ID': 'CustomerID',
    'ProductID': 'StockCode',
    'ItemID': 'StockCode'
}, inplace=True)

required_cols = ['CustomerID','StockCode','Description','Quantity','UnitPrice','InvoiceDate']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

df.dropna(subset=['CustomerID','StockCode','Description'], inplace=True)

df['CustomerID'] = df['CustomerID'].astype(int)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df.dropna(subset=['InvoiceDate'], inplace=True)

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# -----------------------------
# 📊 MENU
# -----------------------------
menu = st.sidebar.selectbox("Select Module", [
    "Dashboard","Data Explorer","RFM Analysis","Segmentation",
    "Collaborative Filtering","Content-Based","Hybrid Recommender",
    "Sales Analytics","Top Products"
])

# -----------------------------
# 📊 Dashboard
# -----------------------------
if menu == "Dashboard":
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", df['CustomerID'].nunique())
    col2.metric("Products", df['StockCode'].nunique())
    col3.metric("Revenue", round(df['TotalPrice'].sum(),2))

# -----------------------------
# 🔍 Data Explorer
# -----------------------------
elif menu == "Data Explorer":
    st.subheader("Dataset")
    st.write(df.head())

    st.subheader("Data Quality")
    st.write(df.isnull().sum())
    st.write("Duplicates:", df.duplicated().sum())

# -----------------------------
# 📊 RFM Analysis
# -----------------------------
elif menu == "RFM Analysis":
    snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency','Frequency','Monetary']

    # Scoring
    rfm['R'] = pd.qcut(rfm['Recency'],4,labels=[4,3,2,1])
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'),4,labels=[1,2,3,4])
    rfm['M'] = pd.qcut(rfm['Monetary'],4,labels=[1,2,3,4])

    rfm['Score'] = rfm[['R','F','M']].astype(str).sum(axis=1)

    st.write(rfm.head())

    csv = rfm.to_csv().encode()
    st.download_button("Download RFM", csv, "rfm.csv")

# -----------------------------
# 👥 Segmentation
# -----------------------------
elif menu == "Segmentation":
    snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency','Frequency','Monetary']

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(scaled)

    fig, ax = plt.subplots()
    sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Cluster'], ax=ax)
    st.pyplot(fig)

# -----------------------------
# 🤖 Collaborative Filtering
# -----------------------------
elif menu == "Collaborative Filtering":
    matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)

    matrix_norm = matrix.subtract(matrix.mean(axis=1), axis=0)

    similarity = cosine_similarity(matrix_norm)
    sim_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

    user = st.number_input("User ID", int(matrix.index.min()), int(matrix.index.max()))

    def recommend(user):
        sim_users = sim_df[user].sort_values(ascending=False)[1:6]
        weighted = np.dot(sim_users.values, matrix.loc[sim_users.index])
        scores = pd.Series(weighted, index=matrix.columns)
        return scores.sort_values(ascending=False).head(5)

    if st.button("Recommend"):
        st.write(recommend(user))

# -----------------------------
# 🧾 Content-Based
# -----------------------------
elif menu == "Content-Based":
    df_cb = df[['StockCode','Description']].drop_duplicates().reset_index(drop=True)

    def clean(text):
        return re.sub(r'[^a-zA-Z ]','', text.lower())

    df_cb['Description'] = df_cb['Description'].apply(clean)

    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    matrix = tfidf.fit_transform(df_cb['Description'])

    sim = cosine_similarity(matrix)
    indices = pd.Series(df_cb.index, index=df_cb['StockCode'])

    code = st.text_input("Enter Product Code")

    if st.button("Get Similar"):
        if code in indices:
            idx = indices[code]
            scores = list(enumerate(sim[idx]))
            scores = sorted(scores, key=lambda x:x[1], reverse=True)[1:6]
            result = df_cb.iloc[[i[0] for i in scores]]
            st.write(result)

# -----------------------------
# 🔥 Hybrid
# -----------------------------
elif menu == "Hybrid Recommender":
    st.subheader("Hybrid Recommendation")

    matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
    sim_users = cosine_similarity(matrix)

    user = st.number_input("User ID", int(matrix.index.min()), int(matrix.index.max()))
    product = st.text_input("Product Code")

    if st.button("Recommend Hybrid"):
        collab = matrix.mean().sort_values(ascending=False).head(5).index.tolist()
        content = df['StockCode'].value_counts().head(5).index.tolist()

        final = list(set(collab + content))
        st.write(final[:5])

# -----------------------------
# 📈 Sales Analytics
# -----------------------------
elif menu == "Sales Analytics":
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    sales = df.groupby('Month')['TotalPrice'].sum()

    fig, ax = plt.subplots()
    sales.plot(ax=ax)
    st.pyplot(fig)

# -----------------------------
# 🔥 Top Products
# -----------------------------
elif menu == "Top Products":
    top = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    st.write(top)

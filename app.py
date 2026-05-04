# =========================================================
# 🛒 ADVANCED E-COMMERCE AI SYSTEM (SINGLE FILE - MODULAR)
# =========================================================

import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

# =========================================================
# ⚙️ CONFIG MODULE
# =========================================================
class Config:
    REQUIRED_COLUMNS = [
        'CustomerID', 'StockCode', 'Description',
        'Quantity', 'UnitPrice', 'InvoiceDate'
    ]
    N_CLUSTERS = 4
    TOP_N = 5


# =========================================================
# 📂 DATA LOADER MODULE
# =========================================================
class DataLoader:
    @staticmethod
    def load(file):
        df = pd.read_csv(file, encoding='ISO-8859-1')
        df.columns = df.columns.str.strip()

        df.rename(columns={
            'Customer ID': 'CustomerID',
            'ProductID': 'StockCode',
            'ItemID': 'StockCode'
        }, inplace=True)

        return df


# =========================================================
# 🧹 PREPROCESSING MODULE
# =========================================================
class Preprocessing:
    @staticmethod
    def clean(df):
        df = df.dropna()

        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

        df['CustomerID'] = df['CustomerID'].astype(int)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

        return df


# =========================================================
# 📊 RFM ANALYSIS MODULE
# =========================================================
class RFM:
    @staticmethod
    def compute(df):
        snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        })

        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        return rfm


# =========================================================
# 🤖 CLUSTERING MODULE
# =========================================================
class Clustering:
    @staticmethod
    def apply(rfm):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(rfm)

        kmeans = KMeans(n_clusters=Config.N_CLUSTERS, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(scaled)

        return rfm


# =========================================================
# 🤝 COLLABORATIVE FILTERING MODULE
# =========================================================
class CollaborativeFiltering:
    def __init__(self, df):
        self.user_matrix = df.pivot_table(
            index='CustomerID',
            columns='StockCode',
            values='Quantity',
            fill_value=0
        )

        sim = cosine_similarity(self.user_matrix)
        self.sim_df = pd.DataFrame(sim,
                                   index=self.user_matrix.index,
                                   columns=self.user_matrix.index)

    def recommend(self, user_id, n=5):
        if user_id not in self.user_matrix.index:
            return []

        sim_users = self.sim_df[user_id].sort_values(ascending=False)[1:6]
        sim_data = self.user_matrix.loc[sim_users.index]

        recs = sim_data.mean().sort_values(ascending=False)
        bought = self.user_matrix.loc[user_id]

        return recs[bought == 0].head(n).index.tolist()


# =========================================================
# 🧾 CONTENT BASED MODULE
# =========================================================
class ContentBased:
    def __init__(self, df):
        self.df_cb = df[['StockCode', 'Description']].drop_duplicates()

        tfidf = TfidfVectorizer(stop_words='english')
        self.matrix = tfidf.fit_transform(self.df_cb['Description'])

        self.sim = cosine_similarity(self.matrix, self.matrix)
        self.indices = pd.Series(self.df_cb.index, index=self.df_cb['StockCode']).drop_duplicates()

    def recommend(self, product_code, n=5):
        if product_code not in self.indices:
            return []

        idx = self.indices[product_code]

        scores = list(enumerate(self.sim[idx].flatten()))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]

        return self.df_cb.iloc[[i[0] for i in scores]]


# =========================================================
# 🧠 HYBRID MODEL MODULE
# =========================================================
class HybridRecommender:
    def __init__(self, collab, content):
        self.collab = collab
        self.content = content

    def recommend(self, user_id, product_code, n=5):
        collab_rec = self.collab.recommend(user_id, n)
        content_rec = self.content.recommend(product_code, n)

        content_list = []
        if len(content_rec) > 0:
            content_list = list(content_rec['StockCode'])

        return list(set(collab_rec + content_list))[:n]


# =========================================================
# 📈 VISUALIZATION MODULE
# =========================================================
class Visualization:
    @staticmethod
    def plot_clusters(rfm):
        fig, ax = plt.subplots()
        ax.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'])
        return fig

    @staticmethod
    def sales_trend(df):
        df['Month'] = df['InvoiceDate'].dt.to_period('M')
        sales = df.groupby('Month')['TotalPrice'].sum()

        fig, ax = plt.subplots()
        sales.plot(ax=ax)
        return fig


# =========================================================
# 🧰 UTIL MODULE
# =========================================================
class Utils:
    @staticmethod
    def validate(df):
        missing = [c for c in Config.REQUIRED_COLUMNS if c not in df.columns]
        return missing


# =========================================================
# 🎯 STREAMLIT APP (MAIN)
# =========================================================
st.set_page_config(page_title="Advanced AI System", layout="wide")
st.title("🛒 Advanced E-Commerce AI System")

file = st.file_uploader("Upload CSV Dataset", type=['csv'])

if file:
    df = DataLoader.load(file)

    missing = Utils.validate(df)
    if missing:
        st.error(f"Missing Columns: {missing}")
        st.stop()

    df = Preprocessing.clean(df)

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", df['CustomerID'].nunique())
    col2.metric("Products", df['StockCode'].nunique())
    col3.metric("Revenue", round(df['TotalPrice'].sum(), 2))

    # RFM + Clustering
    rfm = RFM.compute(df)
    rfm = Clustering.apply(rfm)

    st.subheader("👥 Customer Segments")
    st.write(rfm.head())
    st.pyplot(Visualization.plot_clusters(rfm))

    # Models
    collab = CollaborativeFiltering(df)
    content = ContentBased(df)
    hybrid = HybridRecommender(collab, content)

    # Inputs
    st.subheader("🎯 Recommendation Engine")

    user_id = st.number_input("User ID", value=int(df['CustomerID'].min()))
    product_code = st.text_input("Product Code")

    if st.button("Get Recommendations"):
        recs = hybrid.recommend(user_id, product_code)
        st.success(recs)

    # Sales
    st.subheader("📈 Sales Trend")
    st.pyplot(Visualization.sales_trend(df))

    # Top Products
    st.subheader("🔥 Top Products")
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    st.write(top_products)

    # Download
    st.download_button("Download Clean Data", df.to_csv(index=False), "clean_data.csv")

    st.success("✅ System Running Successfully 🚀")
else:
    st.warning("Upload dataset to begin")

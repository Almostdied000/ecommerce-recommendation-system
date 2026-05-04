import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced E-Commerce AI", layout="wide")

st.title("🛒 Advanced E-Commerce AI System")

# -----------------------------
# 📂 FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Upload dataset to continue")
    st.stop()

df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
df.columns = df.columns.str.strip()

# -----------------------------
# 🧠 AUTO COLUMN FIX
# -----------------------------
column_map = {
    'Customer ID': 'CustomerID',
    'ProductID': 'StockCode',
    'ItemID': 'StockCode'
}
df.rename(columns=column_map, inplace=True)

required_cols = ['CustomerID', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'InvoiceDate']
missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# -----------------------------
# 🧹 CLEANING
# -----------------------------
df.dropna(inplace=True)

df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

df['CustomerID'] = df['CustomerID'].astype(int)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# -----------------------------
# 📊 SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔎 Filters")

min_price, max_price = st.sidebar.slider(
    "Price Range",
    float(df['UnitPrice'].min()),
    float(df['UnitPrice'].max()),
    (float(df['UnitPrice'].min()), float(df['UnitPrice'].max()))
)

df = df[(df['UnitPrice'] >= min_price) & (df['UnitPrice'] <= max_price)]

# -----------------------------
# 📊 KPI DASHBOARD
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Customers", df['CustomerID'].nunique())
col2.metric("Products", df['StockCode'].nunique())
col3.metric("Revenue", f"{round(df['TotalPrice'].sum(), 2)}")

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

# -----------------------------
# 🤖 AUTO CLUSTERING (ELBOW STYLE)
# -----------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

inertia = []
K_range = range(2, 8)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)

optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

st.subheader("👥 Customer Segmentation")
st.write(rfm.head())

# -----------------------------
# 📈 VISUALIZATION
# -----------------------------
fig, ax = plt.subplots()
ax.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'])
st.pyplot(fig)

# -----------------------------
# 🤝 COLLABORATIVE FILTERING
# -----------------------------
user_item = df.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    fill_value=0
)

user_sim = cosine_similarity(user_item)

user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)

# -----------------------------
# 🧾 CONTENT BASED
# -----------------------------
df_cb = df[['StockCode', 'Description']].drop_duplicates()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_cb['Description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df_cb.index, index=df_cb['StockCode']).drop_duplicates()

# -----------------------------
# 🧠 HYBRID RECOMMENDER
# -----------------------------
def hybrid_recommend(user_id, product_code, n=5):
    results = []

    # Collaborative
    if user_id in user_item.index:
        sim_users = user_sim_df[user_id].sort_values(ascending=False)[1:6]
        sim_data = user_item.loc[sim_users.index]
        collab = sim_data.mean().sort_values(ascending=False)
        results.extend(list(collab.head(n).index))

    # Content
    if product_code in indices:
        idx = indices[product_code]
        sim_scores = list(enumerate(cosine_sim[idx].flatten()))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        content = [df_cb.iloc[i[0]]['StockCode'] for i in sim_scores]
        results.extend(content)

    return list(set(results))[:n]

# -----------------------------
# 🎯 USER INPUT
# -----------------------------
st.subheader("🎯 Hybrid Recommendation")

user_id = st.number_input("User ID", value=int(user_item.index.min()))
product_code = st.text_input("Product Code")

if st.button("Get Recommendations"):
    recs = hybrid_recommend(user_id, product_code)
    st.write("Recommended Products:", recs)

# -----------------------------
# 📈 SALES TREND
# -----------------------------
df['Month'] = df['InvoiceDate'].dt.to_period('M')
sales = df.groupby('Month')['TotalPrice'].sum()

fig2, ax2 = plt.subplots()
sales.plot(ax=ax2)
st.pyplot(fig2)

# -----------------------------
# 🔥 TOP PRODUCTS
# -----------------------------
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

st.subheader("🔥 Top Products")
st.write(top_products)

# -----------------------------
# 📥 DOWNLOAD
# -----------------------------
st.download_button(
    "Download Clean Data",
    df.to_csv(index=False),
    file_name="clean_data.csv"
)

st.success("✅ Advanced AI System Running Successfully 🚀")

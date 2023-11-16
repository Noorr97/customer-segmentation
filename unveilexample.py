import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/teamsmcmjcet/Customer-Segmentation/main/Dataset/Mall_Customers.csv")

# Introduction paragraph
st.title("Customer Segmentation Analysis")
st.write("Customer segmentation is a crucial aspect of marketing strategy, allowing businesses to understand and target different customer groups effectively. In this analysis, we use K-Means clustering to categorize customers based on their annual income and spending score.Data science plays a vital role in solving real-world problems. By leveraging techniques like clustering, businesses can gain valuable insights from their data, enabling informed decision-making and tailored strategies for customer engagement and satisfaction.")

# Sidebar with user input
st.sidebar.title("Customer Segmentation Parameters")
cluster_algorithm = st.sidebar.radio("Select Clustering Algorithm", ["KMeans"])

# K-Means Parameters
if cluster_algorithm == "KMeans":
    st.sidebar.header('K-Means Parameters')
    num_clusters = st.sidebar.slider('Select Number of Clusters:', 2, 5, 3)

# Perform clustering based on user selection
if cluster_algorithm == "KMeans":
    x = df.iloc[:, [3, 4]].values
    model = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
    y_kmeans = model.fit_predict(x)
    df['label_kmeans'] = y_kmeans

    # Customize colors and symbols for each cluster
    cluster_colors = ["red", "green", "blue", "purple", "orange"]  # Use a different color palette
    cluster_symbols = ["circle", "square", "diamond", "hexagon", "+"]  # Define your preferred symbols here
    df['cluster_color'] = df['label_kmeans'].map(lambda label: cluster_colors[label % len(cluster_colors)])
    df['cluster_symbol'] = df['label_kmeans'].map(lambda label: cluster_symbols[label % len(cluster_symbols)])

    # Plot 3D Scatter Plot using Plotly Express
    fig_3d = px.scatter_3d(df, x='Annual Income (k$)', y='Spending Score (1-100)', z='Age',
                            color='cluster_color', symbol='cluster_symbol', size_max=18, opacity=0.7, height=1000, width=1000)
    st.plotly_chart(fig_3d)

# Display the raw data
st.subheader('Raw Data')
st.write(df)

# 2D Scatter Plot with K-Means parameters
st.subheader("2D Scatter Plot")
plt.figure(figsize=(8, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['label_kmeans'], cmap='viridis', s=60)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title(f'KMeans Clustering (k={num_clusters})')
st.pyplot()

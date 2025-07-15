import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import io
import base64

def preprocess_data_for_clustering(df, numeric_columns=None):
    """Preprocess data for clustering by selecting and scaling numeric features"""
    # If no specific columns are provided, use all numeric columns
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Drop any non-numeric columns from the list
    valid_columns = [col for col in numeric_columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    # Ensure we have at least 2 columns for clustering
    if len(valid_columns) < 2:
        raise ValueError("Not enough numeric columns for clustering. Need at least 2.")
    
    # Extract features
    features = df[valid_columns].copy()
    
    # Handle missing values if any
    features = features.fillna(features.mean())
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, valid_columns, scaler

def apply_kmeans_clustering(df, n_clusters=3, numeric_columns=None, random_state=42):
    """Apply K-means clustering to the given dataframe"""
    # Preprocess data
    try:
        scaled_features, valid_columns, scaler = preprocess_data_for_clustering(df, numeric_columns)
    except ValueError as e:
        return None, str(e)
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = calculate_cluster_stats(df_with_clusters, valid_columns)
    
    return df_with_clusters, cluster_stats

def calculate_cluster_stats(df_with_clusters, feature_columns):
    """Calculate statistics for each cluster"""
    stats = {}
    
    # Overall statistics
    stats['cluster_sizes'] = df_with_clusters['cluster'].value_counts().sort_index().to_dict()
    stats['total_clusters'] = len(stats['cluster_sizes'])
    
    # Per-cluster statistics
    stats['cluster_means'] = {}
    stats['cluster_ids'] = {}
    
    for cluster_id in range(stats['total_clusters']):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        # Store mean values for each feature in this cluster
        stats['cluster_means'][cluster_id] = {
            col: cluster_data[col].mean() for col in feature_columns
        }
        
        # Store IDs in this cluster
        stats['cluster_ids'][cluster_id] = cluster_data['id'].tolist()
    
    return stats

def get_optimal_clusters(df, max_clusters=10, numeric_columns=None):
    """Determine the optimal number of clusters using the Elbow Method"""
    try:
        scaled_features, _, _ = preprocess_data_for_clustering(df, numeric_columns)
    except ValueError as e:
        return None, str(e)
    
    # Calculate distortion for different numbers of clusters
    distortions = []
    K = range(1, min(max_clusters + 1, len(df) + 1))
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        distortions.append(kmeans.inertia_)
    
    # Create elbow plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K, distortions, 'bx-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Distortion (Inertia)')
    ax.set_title('The Elbow Method showing the optimal k')
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Return the plot as base64 encoded string
    return base64.b64encode(buf.read()).decode(), "Success"

def plot_cluster_comparison(df_with_clusters, x_feature, y_feature):
    """Generate a scatter plot comparing two features with points colored by cluster"""
    if x_feature not in df_with_clusters.columns or y_feature not in df_with_clusters.columns:
        return None, f"Features {x_feature} or {y_feature} not found in the dataframe."
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique clusters
    clusters = df_with_clusters['cluster'].unique()
    
    # Plot each cluster with a different color
    for cluster in clusters:
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
        ax.scatter(
            cluster_data[x_feature],
            cluster_data[y_feature],
            alpha=0.6,
            label=f'Cluster {cluster}'
        )
    
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f'Cluster Comparison: {x_feature} vs {y_feature}')
    ax.legend()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Return the plot as base64 encoded string
    return base64.b64encode(buf.read()).decode(), "Success"

def display_cluster_info(cluster_stats, df_original):
    """Display cluster information in Streamlit"""
    total_customers = len(df_original)
    
    # Display overview
    st.subheader("Customer Segments Overview")
    st.write(f"Total customers: {total_customers}")
    st.write(f"Number of segments: {cluster_stats['total_clusters']}")
    
    # Create a bar chart for cluster sizes
    cluster_sizes = cluster_stats['cluster_sizes']
    cluster_labels = [f"Segment {i+1}" for i in range(len(cluster_sizes))]
    sizes = list(cluster_sizes.values())
    
    # Display the bar chart
    st.bar_chart({
        'Segment': cluster_labels,
        'Customers': sizes
    })
    
    # Display detailed information for each cluster
    st.subheader("Segment Details")
    
    for i in range(cluster_stats['total_clusters']):
        with st.expander(f"Segment {i+1} ({cluster_sizes[i]} customers)"):
            # Display mean values for key metrics
            means = cluster_stats['cluster_means'][i]
            metrics_df = pd.DataFrame({
                'Metric': list(means.keys()),
                'Average Value': list(means.values())
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # Show sample IDs from this cluster
            ids = cluster_stats['cluster_ids'][i]
            sample_size = min(10, len(ids))
            st.write(f"Sample customer IDs: {', '.join(map(str, ids[:sample_size]))}")
            if len(ids) > sample_size:
                st.write(f"... and {len(ids) - sample_size} more")

def get_cluster_data_for_download(df_with_clusters):
    """Prepare cluster data for download as CSV"""
    # Select relevant columns for download
    download_df = df_with_clusters[['id', 'cluster']].copy()
    download_df.columns = ['customer_id', 'segment']
    
    # Rename clusters to start from 1 instead of 0 for user-friendliness
    download_df['segment'] = download_df['segment'] + 1
    
    # Convert to CSV
    csv = download_df.to_csv(index=False)
    
    # Encode to base64
    b64 = base64.b64encode(csv.encode()).decode()
    
    return b64
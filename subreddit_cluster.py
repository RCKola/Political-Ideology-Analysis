
import torch
from partisannet.data.datamodule import get_dataloaders
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from partisannet.models.get_embeddings import generate_embeddings
import numpy as np


def erase(embeddings, eraser=joblib.load("data/svm/linear_eraser.joblib")):
    return eraser(torch.tensor(embeddings)).numpy()

def find_K_cluster(X_erased):
    K_range = range(2, 20) # Test from 2 to 20 clusters
    inertias = []
    silhouettes = []

    print("Searching for optimal K...")

    for k in K_range:
        # Run K-Means
        clustering = AgglomerativeClustering(n_clusters=k, random_state=42)
        labels = clustering.fit_predict(X_erased)
        
        
        inertias.append(clustering.inertia_ if hasattr(clustering, "inertia_") else 0)
        
        score = silhouette_score(X_erased, labels)
        silhouettes.append(score)
        
        print(f"K={k}: Silhouette={score:.3f}")

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Inertia (Elbow)
    ax1.plot(K_range, inertias, 'bo-', label='Inertia (Elbow)')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title("Optimal K Search: Elbow Method vs. Silhouette Score")

    # Plot Silhouette (on secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(K_range, silhouettes, 'rs--', label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.grid(True, alpha=0.3)
    plt.show()

def cluster_subreddits(X_erased, subreddit_centroids, avg_labels, K_CLUSTERS=6):
    clustering = AgglomerativeClustering(n_clusters=K_CLUSTERS, linkage='ward')
    semantic_labels = clustering.fit_predict(X_erased)


    coords_erased = PCA(n_components=2).fit_transform(X_erased)

   
    df_viz = pd.DataFrame({
        'x': coords_erased[:, 0],
        'y': coords_erased[:, 1],
        'cluster': semantic_labels.astype(str), # Convert to string for categorical coloring
        'label': list(subreddit_centroids.keys()) , #list of topic names (e.g. "gun_control")
        'Right_Leaning_Percentage': list(avg_labels.values()) # Percentage of Right-Leaning content in each topic
    })

    plt.figure(figsize=(12, 10))

    # Plot the clusters
    sns.scatterplot(
        data=df_viz, 
        x='x', y='y', 
        hue='cluster', 
        palette='tab10', 
        s=150, 
        alpha=0.9,
        edgecolor='k'
    )

   
    for i in range(df_viz.shape[0]):
        plt.text(
            df_viz.x[i]+0.02, 
            df_viz.y[i], 
            df_viz.label[i], 
            fontsize=9, 
            alpha=0.8
        )

    plt.title(f"Semantic Grouping of Topics After Removing Political Bias (k={K_CLUSTERS})", fontsize=15)
    plt.xlabel("Principal Component 1 (Politics Removed)")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Semantic Cluster")
    plt.grid(True, alpha=0.3)
    plt.show()

   
    print("\n--- Cluster Analysis ---")
    for k in range(K_CLUSTERS):
        cluster_topics = df_viz[df_viz['cluster'] == str(k)]['label'].tolist()
        cluster_percentages = df_viz[df_viz['cluster'] == str(k)]['Right_Leaning_Percentage'].tolist()
        print(f"Cluster {k}: {cluster_topics}")
        print(f"Percentage Right-Leaning:")
        print(cluster_percentages)

    return df_viz # Return the DataFrame for potential further analysis or JSON export

def create_json_table(df_orig, df_erased):
    
    df_orig_subset = df_orig[['label', 'cluster']].copy()
    df_erased_subset = df_erased[['label', 'cluster']].copy()

    df_orig_subset['label'] = df_orig_subset['label'].apply(lambda x: x[0] if isinstance(x, list) else str(x))
    df_erased_subset['label'] = df_erased_subset['label'].apply(lambda x: x[0] if isinstance(x, list) else str(x))

    
    df_orig_subset = df_orig_subset.rename(columns={'cluster': 'original_cluster'})
    df_erased_subset = df_erased_subset.rename(columns={'cluster': 'new_cluster'})

    
    df_combined = pd.merge(df_orig_subset, df_erased_subset, on='label')

   
    df_combined = df_combined.rename(columns={'label': 'topic_name'})

    df_combined['original_cluster'] = df_combined['original_cluster'].astype(int)
    df_combined['new_cluster'] = df_combined['new_cluster'].astype(int)

    # 6. Export to JSON
    df_combined.to_json("Results/subreddit_cluster_comparison.json", orient="records", indent=4)



if __name__ == "__main__":
    
    find_K = False  # Set to True to run the K search, False to just run with a fixed K (e.g., 6)


    dataloaders = get_dataloaders("subreddits", batch_size=32, split=False, renew_cache=False)
    embeddings, partisan_labels, subreddits = generate_embeddings(dataloaders['train'], path = "data/centerloss_sbert_full")
    
    cf_svm = joblib.load("data/svm/svm_model.joblib")
    labels = cf_svm.predict(embeddings)
    
    df =  pd.DataFrame({"embedding": list(embeddings), "subreddit": subreddits, "label": labels})
    


    
    subreddit_centroids = df.groupby('subreddit')['embedding'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).to_dict()
    avg_labels = df.groupby('subreddit')['label'].apply(
        lambda x: np.mean(x)
    ).to_dict()

    X_erased = normalize(erase(list(subreddit_centroids.values())),norm='l2')

    

    if find_K:
        find_K_cluster(X_erased)
    
    
    
    K_CLUSTERS = 6 # We set this based on the K search results and choose a value that balances granularity with interpretability

    print(f"Clustering erased centroids into {K_CLUSTERS} semantic groups...")

    df_erased = cluster_subreddits(X_erased, subreddit_centroids, avg_labels, K_CLUSTERS=K_CLUSTERS) # Erase, cluster, and visualize the subreddits

    X_orig = normalize(list(subreddit_centroids.values()),norm='l2')

    df_orig = cluster_subreddits(X_orig, subreddit_centroids, avg_labels, K_CLUSTERS=K_CLUSTERS) # Cluster and visualize the original (non-erased) centroids for comparison

    create_json_table(df_orig, df_erased)
    

 
    

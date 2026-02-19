
import torch
from concept_erasure import LeaceEraser
from partisannet.data.datamodule import get_dataloaders
from sklearn.cluster import AgglomerativeClustering, KMeans
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from partisannet.models.get_embeddings import generate_embeddings
import numpy as np




def create_eraser(embeddings):
    svm = joblib.load("data/svm/svm_model.joblib")
    labels = svm.predict(embeddings)
    eraser = LeaceEraser.fit(torch.tensor(embeddings), torch.tensor(labels))
    joblib.dump(eraser, "data/svm/linear_eraser.joblib")
    return eraser

def erase(embeddings, eraser=joblib.load("data/svm/linear_eraser.joblib")):
    return eraser(torch.tensor(embeddings)).numpy()


if __name__ == "__main__":
    train = False
    find_K = False
    if train:
        dataloaders = get_dataloaders("topic_data", batch_size=32, split=False, renew_cache=False)
        embeddings, partisan_labels, _ = generate_embeddings(dataloaders['train'], path = "data/centerloss_sbert_full")
        create_eraser(embeddings)

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

    X_erased = normalize(list(subreddit_centroids.values()),norm='l2')

    

    if find_K:
        K_range = range(2, 20) # Test from 2 to 20 clusters
        inertias = []
        silhouettes = []

        print("Searching for optimal K...")

        for k in K_range:
            # Run K-Means
            clustering = KMeans(n_clusters=k, random_state=42)
            labels = clustering.fit_predict(X_erased)
            
            # 1. Inertia (Sum of squared distances to center) -> For Elbow Method
            inertias.append(clustering.inertia_ if hasattr(clustering, "inertia_") else 0) # AgglomerativeClustering doesn't have inertia, so we set it to 0 or you can compute it manually    
            # 2. Silhouette Score (How distinct are the clusters?) -> Maximize this
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
        exit()

    K_CLUSTERS = 6

    print(f"Clustering erased centroids into {K_CLUSTERS} semantic groups...")

    # 2. Fit K-Means on the CLEANED (Erased) vectors
    clustering = AgglomerativeClustering(n_clusters=K_CLUSTERS, linkage='ward')
    semantic_labels = clustering.fit_predict(X_erased)

    # 3. Visualization
    # We use the same PCA coordinates from the previous step to keep the "map" consistent
    # If you lost coords_erased, uncomment the line below:
    coords_erased = PCA(n_components=2).fit_transform(X_erased)

    # Create a DataFrame for easier plotting with names
    df_viz = pd.DataFrame({
        'x': coords_erased[:, 0],
        'y': coords_erased[:, 1],
        'cluster': semantic_labels.astype(str), # Convert to string for categorical coloring
        'label': list(subreddit_centroids.keys()) , # Your list of topic names (e.g. "gun_control")
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

    # 4. Annotate points to check the semantics
    # We verify if the cluster makes sense (e.g., does Cluster 0 contain both Tax and Wage?)
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

    # 5. Print the text content of clusters for your paper
    print("\n--- Cluster Analysis ---")
    for k in range(K_CLUSTERS):
        cluster_topics = df_viz[df_viz['cluster'] == str(k)]['label'].tolist()
        cluster_percentages = df_viz[df_viz['cluster'] == str(k)]['Right_Leaning_Percentage'].tolist()
        print(f"Cluster {k}: {cluster_topics}")
        print(f"Percentage Right-Leaning:")
        print(cluster_percentages)



import torch
from concept_erasure import LeaceEraser
from partisannet.data.datamodule import get_dataloaders
from partisannet.data.topicmodule import load_topic_model
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from partisannet.models.get_embeddings import generate_embeddings





def create_eraser(embeddings):
    svm = joblib.load("data/svm/svm_model.joblib")
    labels = svm.predict(embeddings)
    eraser = LeaceEraser.fit(torch.tensor(embeddings), torch.tensor(labels))
    joblib.dump(eraser, "data/svm/linear_eraser.joblib")
    return eraser

def erase(embeddings, eraser=joblib.load("data/svm/linear_eraser.joblib")):
    return eraser(torch.tensor(embeddings)).numpy()


if __name__ == "__main__":
    train = True # Set to True to train the eraser, False to just load and use it
    find_K = False # Set to True to run the K search, False to just run with a fixed K (e.g., 30)
    if train:
        dataloaders = get_dataloaders("topic_data", batch_size=32, split=False, renew_cache=False)
        embeddings, partisan_labels, _ = generate_embeddings(dataloaders['train'], path = "data/centerloss_sbert_full")
        create_eraser(embeddings)

    topic_model, X_centroids, topic_labels, topic_percentages = load_topic_model("topic_data", embedding_model="data/centerloss_sbert_full", renew_cache=False, num_topics=500, cluster_in_k=None)
    print(X_centroids.shape)
    

    X_erased = normalize(erase(X_centroids),norm='l2')
    pca = PCA(n_components=50)
    X_reduced = pca.fit_transform(X_erased)
    

    if find_K:
        K_range = range(2, 64) # Test from 2 to 64 clusters
        inertias = []
        silhouettes = []

        print("Searching for optimal K...")

        for k in K_range:
            # Run K-Means
            clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = clustering.fit_predict(X_reduced)
            
            
            inertias.append(clustering.inertia_ if hasattr(clustering, "inertia_") else 0) 
            
            score = silhouette_score(X_reduced, labels)
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

    K_CLUSTERS = 20

    print(f"Clustering erased centroids into {K_CLUSTERS} semantic groups...")


    clustering = AgglomerativeClustering(n_clusters=K_CLUSTERS, linkage='ward')
    semantic_labels = clustering.fit_predict(X_reduced)


    coords_erased = PCA(n_components=2).fit_transform(X_erased)


    df_viz = pd.DataFrame({
        'x': coords_erased[:, 0],
        'y': coords_erased[:, 1],
        'cluster': semantic_labels.astype(str), # Convert to string for categorical coloring
        'label': topic_labels, #list of topic names (e.g. "gun_control")
        'Right_Leaning_Percentage': topic_percentages # Percentage of Right-Leaning content in each topic
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


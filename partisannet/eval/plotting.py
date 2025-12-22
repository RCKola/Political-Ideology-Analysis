import matplotlib.pyplot as plt
import numpy as np

def plot_umap(embeddings, labels):
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='coolwarm', s=10, alpha=0.7)
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Left (0)', 'Right (1)'])
    
    plt.title("UMAP of Binary Partisan Embeddings (Fine-Tuned)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    output_file = "umap_binary.png"
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()
    plt.close()

def plot_phate(embeddings, labels):
    import phate
    phate_op = phate.PHATE(n_components=2, knn=5, decay=40, n_jobs=-1)
    data_phate = phate_op.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_phate[:, 0], data_phate[:, 1], c=labels, cmap='coolwarm', s=5, alpha=0.7)
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Left (0)', 'Right (1)'])

    plt.title("PHATE Visualization of Embeddings")
    plt.tight_layout()
    output_file ="phate.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()
    plt.close()

def plot_density_distrib(embeddings, labels, k, caption=None):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)

    # Count the density distribution for each class in each cluster
    cluster_counts = {cluster: [0, 0] for cluster in range(k)}
    for label, cls in zip(kmeans_labels, labels):
        cluster_counts[label][cls.item()] += 1

    count1 = [cluster_counts[i][0] for i in range(k)]
    count2 = [cluster_counts[i][1] for i in range(k)]

    import seaborn as sns
    sns.set_style("whitegrid")
    x = np.arange(k)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bar1 = ax.bar(x - width/2, count1, width, label="Left", color='midnightblue')
    bar2 = ax.bar(x + width/2, count2, width, label="Right", color='darkorange')

    ax.set_xlabel('Clusters')
    ax.set_ylabel('Density Count')
    ax.set_title('Cluster Density Distribution by Source Before Erasure', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(k)], rotation=45)
    ax.legend()

    # Save the plot
    plt.tight_layout()
    output_file = f"KMeans_{caption}.png" if caption is not None else "KMeans.png"
    plt.savefig(output_file)
    plt.close()

def plot_topic_distribution(topic_model, topics, partisan_labels, report=False):
    import torch
    topics_np = topics.cpu().numpy() if isinstance(topics, torch.Tensor) else topics
    labels_np = partisan_labels.cpu().numpy() if isinstance(partisan_labels, torch.Tensor) else partisan_labels

    import pandas as pd
    df = pd.DataFrame({'Topic_ID': topics_np,'Ideology': labels_np})

    # Create the pivot table
    topic_stats = df.pivot_table(index='Topic_ID', columns='Ideology', aggfunc='size', fill_value=0)
    topic_stats.columns = ['Left_Count', 'Right_Count'] 

    # Calculate new columns (Total and Percentages)
    topic_stats['Total'] = topic_stats['Left_Count'] + topic_stats['Right_Count']
    topic_stats['%_Left'] = (topic_stats['Left_Count'] / topic_stats['Total'] * 100).round(1)
    topic_stats['%_Right'] = (topic_stats['Right_Count'] / topic_stats['Total'] * 100).round(1)

    topic_names = topic_model.get_topic_info()[['Topic', 'Name']]
    topic_stats = topic_stats.merge(topic_names, left_on='Topic_ID', right_on='Topic', how='left')

    topic_stats['Name'] = topic_stats['Name'].apply(lambda x: "_".join(x.split("_")[1:]))
    topic_stats = topic_stats[['Topic', 'Name', 'Left_Count', 'Right_Count', '%_Left', '%_Right', 'Total']]

    topic_stats = topic_stats.sort_values('Total', ascending=False)
    print(topic_stats)

    if report:
        topic_stats.to_csv("topic_breakdown.csv")
    
    # Plot a stacked bar chart
    topic_stats[['Left_Count', 'Right_Count']].plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title("Topic Distribution by Political Leaning")
    plt.ylabel("Number of Embeddings")
    plt.xlabel("Topic ID")
    plt.show() 
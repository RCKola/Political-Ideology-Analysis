import os
import torch
from concept_erasure import LeaceEraser
from partisannet.data.datamodule import get_dataloaders
from partisannet.models.classifier import SBERTClassifier
from partisannet.data.datamodule import include_topics, load_datasets
from partisannet.modelling import PartisanNetModel 
from partisannet.models.get_embeddings import generate_embeddings
from datasets import load_from_disk

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def is_top_k_match(embeddings_0, embeddings_1, k=3):
    
    # Combine both lists into a single array
    all_embeddings = np.concatenate((embeddings_0, embeddings_1), axis=0)  # shape (512, embedding_size)
    num_cases = len(embeddings_0)
    top_k_matches = []

    for i in range(num_cases):
        case_embedding = embeddings_0[i].reshape(1, -1)  # Reshape for the cosine_similarity function

        # Calculate cosine similarity between this case embedding and all embeddings
        similarities = cosine_similarity(case_embedding, all_embeddings).flatten()

        # Get the similarity score with the corresponding oral argument
        corresponding_similarity = similarities[num_cases + i]  # index of the corresponding oral argument

        # Find the top k similarity scores (excluding the case embedding itself)
        top_k_indices = np.argsort(similarities)[-(k+1):-1]  # Get the indices of the top 3 similarities
        top_k_similarities = similarities[top_k_indices]  # Corresponding similarity scores

        # Check if the corresponding similarity is in the top k
        is_top_k = corresponding_similarity >= np.min(top_k_similarities)
        top_k_matches.append(is_top_k)

    return top_k_matches

def get_exact_pairs(X, k, labels_mask0):

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Determine cluster labels for source_1 and source_2 embeddings
    labels_1 = clusters[labels_mask0]
    labels_2 = clusters[labels_mask0 == False]

    # Count exact matches (source_1, source_2 pairs) in each cluster
    matches_per_cluster = Counter()

    for i, label_1 in enumerate(labels_1):
        # Check if the corresponding embedding_2 has the same cluster label
        if label_1 == labels_2[i]:
            matches_per_cluster[label_1] += 1

    # Print the number of matches per cluster
    total_pairs = 0
    for cluster_id in range(k):
        total_pairs += matches_per_cluster[cluster_id]

    return total_pairs

def run_erasure_two_sources(
        embeddings,
        embeddings_erased,
        topic_labels,
        k=5, 
        top_k_retrieval=20, 
        max_n_clusters=32,
        source_1 = "has topic",
        source_2 = "outlier"
    ): 
    numeric_labels = topic_labels.clone()
    label_mask0 = (topic_labels == 0).numpy()

    ###############################################
    ### Run k-means clustering - before erasure ###
    ###############################################
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings.numpy())

    # Count the density distribution for each source in each cluster
    cluster_counts = {cluster: [0, 0] for cluster in range(k)}
    for label, source in zip(kmeans_labels, numeric_labels):
        cluster_counts[label][source.item()] += 1

    source_1_counts = [cluster_counts[i][0] for i in range(k)]
    source_2_counts = [cluster_counts[i][1] for i in range(k)]

    # Plotting 
    sns.set_style("whitegrid")
    x = np.arange(k)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bar1 = ax.bar(x - width/2, source_1_counts, width, label=source_1, color='midnightblue')
    bar2 = ax.bar(x + width/2, source_2_counts, width, label=source_2, color='darkorange')

    ax.set_xlabel('Clusters')
    ax.set_ylabel('Density Count')
    ax.set_title('Cluster Density Distribution by Source Before Erasure', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(k)], rotation=45)
    ax.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig('kmeans_before_erasure.png')
    plt.close()

    ##############################################
    ### Run k-means clustering - after erasure ###
    ##############################################
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings_erased.numpy())

    # Count the density distribution for each source in each cluster
    cluster_counts = {cluster: [0, 0] for cluster in range(k)}
    for label, source in zip(kmeans_labels, numeric_labels):
        cluster_counts[label][source.item()] += 1

    source_1_counts = [cluster_counts[i][0] for i in range(k)]
    source_2_counts = [cluster_counts[i][1] for i in range(k)]

    # Plotting
    x = np.arange(k)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bar1 = ax.bar(x - width/2, source_1_counts, width, label=source_1, color='midnightblue')
    bar2 = ax.bar(x + width/2, source_2_counts, width, label=source_2, color='darkorange')

    ax.set_xlabel('Clusters')
    ax.set_ylabel('Density Count')
    ax.set_title('Cluster Density Distribution by Source After Erasure', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(k)], rotation=45)
    ax.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig('kmeans_after_erasure.png')
    plt.close()

    ####################################
    ### Perform PCA - before erasure ###
    ####################################
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # Separate PCA results by source
    source_1_pca = pca_result[label_mask0]
    source_2_pca = pca_result[label_mask0 == False]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(source_1_pca[:, 0], source_1_pca[:, 1], color='midnightblue', label=source_1, alpha=0.7, s=15)  # Smaller points
    plt.scatter(source_2_pca[:, 0], source_2_pca[:, 1], color='darkorange', label=source_2, alpha=0.7, s=15)  # Smaller points
    plt.title('PCA of Embeddings Before Erasure', fontsize=20)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('pca_before_erasure.png')
    plt.close()

    ###################################
    ### Perform PCA - after erasure ###
    ###################################
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings_erased)

    # Separate PCA results by source
    source_1_pca = pca_result[label_mask0]
    source_2_pca = pca_result[label_mask0 == False]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(source_1_pca[:, 0], source_1_pca[:, 1], color='midnightblue', label=source_1, alpha=0.7, s=15)  # Smaller points
    plt.scatter(source_2_pca[:, 0], source_2_pca[:, 1], color='darkorange', label=source_2, alpha=0.7, s=15)  # Smaller points
    plt.title('PCA of Embeddings After Erasure', fontsize=20)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('pca_after_erasure.png')
    plt.close()

    #########################
    ### Get Top-K Ranking ###
    #########################
    
    # Split back into two sets 
    embeddings_1 = embeddings[label_mask0]
    embeddings_2 = embeddings[label_mask0 == False]
    embeddings_erased_1 = embeddings_erased[label_mask0]
    embeddings_erased_2 = embeddings_erased[label_mask0 == False]
    
    ret_list, ret_list_erased = [], []
    for tk in range(top_k_retrieval, 0, -1):

        top_k_results = is_top_k_match(embeddings_1, embeddings_2, k=tk)
        prc = sum(top_k_results) / len(top_k_results)
        ret_list.append(prc)

        top_k_results = is_top_k_match(embeddings_erased_1, embeddings_erased_2, k=tk)
        prc = sum(top_k_results) / len(top_k_results)
        ret_list_erased.append(prc)

    # X-axis labels
    x_labels = [f"{i+1}" for i in range(len(ret_list))[::-1]]

    # Plot each list
    plt.figure(figsize=(10.5, 6))
    plt.plot(x_labels, ret_list, marker='o', color='midnightblue', linestyle='-', label='Before')
    plt.plot(x_labels, ret_list_erased, marker='^', color='darkorange', linestyle='-', label='After')
    
    # Adding labels and legend
    plt.title('Retrieval Before vs. After LEACE', fontsize=20)
    plt.ylabel('Percentage', fontsize=12)
    plt.xlabel('Top k', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Add grid for better readability

    # Save the plot
    plt.tight_layout()
    plt.savefig('top_k_retrieval.png')
    plt.close()

    ### Get exact pairs

    total_pairs, total_pairs_erased = [], []

    k_list = list(range(2, max_n_clusters))

    for kr in k_list:
        total_pairs.append(get_exact_pairs(embeddings, kr, label_mask0))
        total_pairs_erased.append(get_exact_pairs(embeddings_erased, kr, label_mask0))

    ln = len(embeddings_1)
    total_pairs_prc = [i / ln for i in total_pairs][:max_n_clusters-1]
    total_pairs_erased_prc = [i / ln for i in total_pairs_erased][:max_n_clusters-1]

    # Create the plot
    plt.figure(figsize=(10.5, 6))
    plt.plot(k_list, total_pairs_prc, marker='.', label='Before Erasure', color='midnightblue')
    plt.plot(k_list, total_pairs_erased_prc, marker='.', label='After Erasure', color='darkorange')

    # Add labels, title, and legend
    plt.xlabel('# of Clusters', fontsize=14)
    plt.ylabel('Percentage of Exact Pairs', fontsize=14)
    plt.title('Percentage of Exact Pairs Before and After Erasure', fontsize=20)
    plt.legend(fontsize=12)

    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.tight_layout()
    plt.savefig('exact_pairs.png')
    plt.close()

if __name__ == "__main__":
    #show_topics()
    #cache_path = "./cached_processed_dataset"
    embeddings_cache_path = "./cached_embeddings.pt"
    partisan_labels_cache_path = "./cached_partisan_labels.pt"
    topics_cache_path = "./cached_topics.pt"
    renew_cache = False
    # 2. Check if cache exists
    if os.path.exists(partisan_labels_cache_path) and os.path.exists(topics_cache_path) and os.path.exists(embeddings_cache_path) and not renew_cache:
        print("Loading dataset and embeddings from disk...")
        #dataset = load_from_disk(cache_path)
        embeddings = torch.load(embeddings_cache_path)
        partisan_labels = torch.load(partisan_labels_cache_path)
        topics = torch.load(topics_cache_path)
    else:
        print("Cache not found. Processing data...")
        # --- Your Expensive Logic Here ---
        dataloaders, topic_model = get_dataloaders("LibCon", batch_size=32, split=False, num_topics=5)
        
        # Generate Embeddings
        embeddings, partisan_labels, topics = generate_embeddings(dataloaders['train'])
        
        # Process Dataset (Map/Filter)
        #dataset = load_datasets("LibCon")
        #dataset, topic_model = include_topics(dataset, remove_stopwords=True)
        
        # --- Save to Disk for Next Time ---
        print("Saving dataset and embeddings to disk...")
        #dataset.save_to_disk(cache_path)
        torch.save(embeddings, embeddings_cache_path)
        torch.save(partisan_labels, partisan_labels_cache_path)
        torch.save(topics, topics_cache_path)


    print(f"Extracted embeddings shape: {embeddings.shape}")
  
    topics_label = (topics == -1).long()
    
    eraser = LeaceEraser.fit(embeddings, topics_label)
    embeddings_erased = eraser(embeddings)
    print(f"Erased embeddings shape: {embeddings_erased.shape}")

    run_erasure_two_sources(
        embeddings=embeddings, 
        embeddings_erased=embeddings_erased,
        topic_labels=topics_label,
        k=5,
        top_k_retrieval=20,
        max_n_clusters=32
    )
        
    
"""     print(dataset.column_names)
    print("Topic Info:", topic_model.get_topic_info())
    print("Topic labels:", topic_model.generate_topic_labels(nr_words=1))
    print("Sample topics from dataset:")
    for i in range(5):
        print(f"Doc: {dataset[i]['text'][:50]}... Topic: {dataset[i]['topic']}, Prob: {dataset[i]['topic_prob']}") """
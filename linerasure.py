import os
import torch
from concept_erasure import LeaceEraser
from partisannet.data.datamodule import get_dataloaders
from partisannet.models.classifier import SBERTClassifier
from partisannet.data.datamodule import include_topics, load_datasets
from partisannet.modelling import PartisanNetModel 
from partisannet.models.get_embeddings import generate_embeddings, get_finetuned_embeddings
from datasets import load_from_disk

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from partisannet.models.get_embeddings import get_finetuned_embeddings
from datasets import Dataset


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
        source_1 = "left",
        source_2 = "right"
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

def  svm_erasure(
        embeddings,
        embeddings_erased,
        topic_labels,
        labels,

):
    embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(embeddings.numpy(), labels.numpy(), test_size=0.2, random_state=42)
    embeddings_erased_train, embeddings_erased_test, _, _ = train_test_split(embeddings_erased.numpy(), labels.numpy(), test_size=0.2, random_state=42)

    ros = RandomOverSampler(random_state=42)
    embeddings_resampled, labels_resampled = ros.fit_resample(embeddings_train, labels_train)
    embeddings_erased_resampled, labels_erased_resampled = ros.fit_resample(embeddings_erased_train, labels_train)

    clf_tot = SVC(kernel='linear', random_state=42, C=2.0, class_weight='balanced')
    clf_tot.fit(embeddings_resampled, labels_resampled)
    predictions_tot = clf_tot.predict(embeddings_test)

    print("Classification Report Before Erasure:")
    print(classification_report(labels_test, predictions_tot))

    accuracy_tot = accuracy_score(labels_test, predictions_tot)
    print(f"Accuracy Before Erasure: {accuracy_tot:.4f}")

    clf_era = SVC(kernel='linear', random_state=42, C=2.0, class_weight='balanced')
    clf_era.fit(embeddings_erased_resampled, labels_erased_resampled) 

    predictions_era = clf_era.predict(embeddings_erased_test)
    print("Classification Report After Erasure:")
    print(classification_report(labels_test, predictions_era))

    accuracy_era = accuracy_score(labels_test, predictions_era)
    print(f"Accuracy After Erasure: {accuracy_era:.4f}")

def plot_topic_distribution(topic_model, dataset, topics, partisan_labels):
        # 1. Convert tensors to numpy (if they aren't already)
        # We use .cpu() just in case they are on the GPU
        topics_np = topics.cpu().numpy() if isinstance(topics, torch.Tensor) else topics
        labels_np = partisan_labels.cpu().numpy() if isinstance(partisan_labels, torch.Tensor) else partisan_labels

        # 2. Create a DataFrame
        df = pd.DataFrame({
            'Topic_ID': topics_np,
            'Ideology': labels_np
        })

        # 3. Create the Pivot Table
        # This counts occurrences of each ideology per topic
        # 0 = Left (usually), 1 = Right (usually)
        # 1. Create the Pivot Table
        # This creates columns named 0 and 1 (based on your ideology labels)
        topic_stats = df.pivot_table(
            index='Topic_ID', 
            columns='Ideology', 
            aggfunc='size', 
            fill_value=0
        )

        # 2. Rename columns IMMEDIATELY
        # Adjust this mapping if 0 is Right and 1 is Left in your data
        topic_stats.columns = ['Left_Count', 'Right_Count'] 

        # 3. Calculate new columns (Total and Percentages)
        topic_stats['Total'] = topic_stats['Left_Count'] + topic_stats['Right_Count']
        topic_stats['%_Left'] = (topic_stats['Left_Count'] / topic_stats['Total'] * 100).round(1)
        topic_stats['%_Right'] = (topic_stats['Right_Count'] / topic_stats['Total'] * 100).round(1)

        # 4. Get Topic Names and Merge
        topic_names = topic_model.get_topic_info()[['Topic', 'Name']]
        topic_stats = topic_stats.merge(topic_names, left_on='Topic_ID', right_on='Topic', how='left')

        # 5. Clean Topic Names
        topic_stats['Name'] = topic_stats['Name'].apply(lambda x: "_".join(x.split("_")[1:]))

        # 6. NOW you can safely reorder/select columns
        # Because they all exist now
        topic_stats = topic_stats[['Topic', 'Name', 'Left_Count', 'Right_Count', '%_Left', '%_Right', 'Total']]

        # 7. Sort and Display
        topic_stats = topic_stats.sort_values('Total', ascending=False)
        print(topic_stats)

        # Optional: Save to CSV for your report
        # topic_stats.to_csv("topic_breakdown.csv")
        

        # Plot a stacked bar chart
        topic_stats[['Left_Count', 'Right_Count']].plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title("Topic Distribution by Political Leaning")
        plt.ylabel("Number of Embeddings")
        plt.xlabel("Topic ID")
        plt.show()
        



if __name__ == "__main__":
    #show_topics()
    #cache_path = "./cached_processed_dataset"
    single_shot = False
    trained_embeddings = True


    dataloaders, topic_model = get_dataloaders("LibCon", batch_size=32, split=False, num_topics=None, cluster_in_k=40, renew_cache=False)
    
    
    if not trained_embeddings:
    # Generate Embeddings
        embeddings, partisan_labels, topics = generate_embeddings(dataloaders['train'])
    else:
        print("Loading pre-computed embeddings from disk...")
        embeddings, partisan_labels, old_topics = generate_embeddings(dataloaders['train'], path = "data/fine_tuned_sbert")

        print("Including topics...")
        temp_ds = Dataset.from_dict({'text': dataloaders['train'].dataset['text']})
        ds_with_topics, topic_model = include_topics(temp_ds, num_topics=None, cluster_in_k=40, embeddings=embeddings.cpu().numpy())

        # Extract the topic IDs
        topics_list = ds_with_topics["topic"]
        topics = torch.tensor(topics_list).long().to(embeddings.device)
       

    print(f"Extracted embeddings shape: {embeddings.shape}")
    print(f"Extracted topics: {topics.shape}")

    

    # 1. Setup Data
    # Ensure these are PyTorch tensors
    # embeddings: shape (N, D)
    # reduced_topics: shape (N, ) with values 0-9 (from reduce_topics)
    current_embeddings = embeddings.clone() 
    unique_topics = torch.unique(topics).tolist()

    if not single_shot:
   
        print(f"Starting iterative erasure for {len(unique_topics)} topics...")

        # 2. Iterative Loop
        for topic_id in unique_topics:
            # A. Create Binary Labels for "Topic X vs. Everything Else"
            # We use (topics == topic_id) to create a boolean mask, then .long() for 0/1 integers
            binary_labels = (topics == topic_id).long()
            
            # B. Fit Eraser on the CURRENT embeddings
            # Note: We fit on 'current_embeddings', which might have already had Topic 0, 1, etc. removed
            eraser = LeaceEraser.fit(current_embeddings, binary_labels)
            
            # C. Apply Erasure
            current_embeddings = eraser(current_embeddings)
            
            print(f"-> Erased info for Topic {topic_id}") 

        # 3. Final Result
        embeddings_erased = current_embeddings
        print("Done! All topics sequentially erased.")
    
    
    else:
        # Single-shot erasure for all topics at once
        print(f"Starting single-shot erasure for {len(unique_topics)} topics...")
        eraser = LeaceEraser.fit(embeddings, topics)
        embeddings_erased = eraser(embeddings)
        
    print(topic_model.get_topic(0)[:3])
    print(f"Erased embeddings shape: {embeddings_erased.shape}")

    plot_topic_distribution(topic_model, dataloaders['train'].dataset, topics, partisan_labels)

    from sklearn.metrics import adjusted_rand_score

    # Move tensors to CPU for scikit-learn
    old_cpu = old_topics.cpu().numpy() if hasattr(old_topics, 'cpu') else old_topics
    new_cpu = topics.cpu().numpy() if hasattr(topics, 'cpu') else topics

    # Calculate similarity
    score = adjusted_rand_score(old_cpu, new_cpu)

    print(f"Topic Similarity Score (ARI): {score:.4f}")


    svm_erasure(
        embeddings=embeddings, 
        embeddings_erased=embeddings_erased,
        topic_labels=unique_topics,
        labels=partisan_labels
    )
    run_erasure_two_sources(
        embeddings=embeddings,
        embeddings_erased=embeddings_erased,
        topic_labels=partisan_labels,
        k=10,
        top_k_retrieval=20,
        max_n_clusters=32,
        source_1 = "left",
        source_2 = "right")
    
"""     print(dataset.column_names)
    print("Topic Info:", topic_model.get_topic_info())
    print("Topic labels:", topic_model.generate_topic_labels(nr_words=1))
    print("Sample topics from dataset:")
    for i in range(5):
        print(f"Doc: {dataset[i]['text'][:50]}... Topic: {dataset[i]['topic']}, Prob: {dataset[i]['topic_prob']}") """
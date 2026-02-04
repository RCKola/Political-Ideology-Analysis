
import torch
from concept_erasure import LeaceEraser
from partisannet.data.datamodule import get_dataloaders
from partisannet.data.topicmodule import load_topic_model, predict_topics
from partisannet.models.get_embeddings import generate_embeddings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from datasets import Dataset
from sklearn.metrics import adjusted_rand_score



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

def svm_report(
        embeddings_train,
        labels_train,
        labels_test,
        embeddings_test,
        clf_tot = SVC(kernel='linear', random_state=42)
):
    clf_tot.fit(embeddings_train, labels_train)
    predictions_tot = clf_tot.predict(embeddings_test)

    print("Classification Report")
    print(classification_report(labels_test, predictions_tot))

    accuracy_tot = accuracy_score(labels_test, predictions_tot)
    print(f"Accuracy: {accuracy_tot:.4f}")

def  svm_test(
        embeddings,
        labels
):
    
    print("Whole Sample")
    embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    svm_report(
        embeddings_train,
        labels_train,
        labels_test,
        embeddings_test
    )

def umap_plot(embeddings_erased, topic_labels, title=""):
    
    # Assume 'embeddings_erased' is your (N, 768) numpy array after erasure

    # 1. FIND TOPICS (Clustering)
    # We guess there might be ~10 main topics. You can also use HDBScan for auto-detection
    # 1. Initialize
    # 2. REDUCE DIMENSIONS (UMAP)
    reducer = umap.UMAP(
        n_neighbors=15,  # 15 is standard; larger (e.g. 50) preserves more global structure
        min_dist=0.1,    # Controls how tightly points are packed
        n_components=2,  # 2D for plotting
        random_state=42
    )
    umap_coords = reducer.fit_transform(embeddings_erased)

    # 3. PLOT
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        umap_coords[:, 0], 
        umap_coords[:, 1], 
        c=topic_labels, 
        cmap='tab20',    # Distinct colors
        s=1,             # Small dot size
        alpha=0.6        # Transparency helps see density
    )

    plt.title(title)
    plt.colorbar(scatter, label="Topic Cluster ID")
    plt.savefig("Plots/"+title.replace(" ", "_") + ".png")
    plt.show()

if __name__ == "__main__":
    #show_topics()
    #cache_path = "./cached_processed_dataset"
    single_shot = True
    trained_embeddings = True
    test = True

    centroid_test = True
    
    topic_model, X_centroids, topic_labels = load_topic_model("topic_data", embedding_model="data/centerloss_sbert", renew_cache=False, num_topics=30, cluster_in_k=None)
    
    if centroid_test:
        from sentence_transformers import SentenceTransformer
        
        derived_direction = np.load('data/cached_data/political_axis.npy')
        direction = derived_direction / np.linalg.norm(derived_direction)

        def remove_component(embeddings, direction):
            projections = (embeddings @ direction.T).reshape(-1, 1) * direction
            return embeddings - projections

        X_erased = remove_component(X_centroids, direction)

        # --- 4. Plotting (PCA is usually better than UMAP for centroids) ---
        from sklearn.decomposition import PCA

        # Fit PCA on the ORIGINAL to define the axes, then transform both
        # This keeps the "view" consistent so you can see them move
        pca = PCA(n_components=2)
        coords_orig = pca.fit_transform(X_centroids)
        coords_erased = pca.transform(X_erased) # Use same axes!

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot A: Original Centroids
        # Color by their position on the political axis (X-axis usually captures variance)
        sns.scatterplot(
            x=coords_orig[:, 0], y=coords_orig[:, 1], 
            s=100, hue=coords_orig[:, 0], palette="coolwarm", legend=False, ax=axes[0]
        )
        # Label a few important topics
        for i, txt in enumerate(topic_labels):
            if i % 5 == 0: # Label every 5th topic to avoid clutter
                axes[0].text(coords_orig[i,0]+0.02, coords_orig[i,1], txt, fontsize=9)

        axes[0].set_title("Original Topic Centroids (Politicized)")
        axes[0].set_xlabel("Principal Component 1 (Likely Political)")

        # Plot B: Erased Centroids
        sns.scatterplot(
            x=coords_erased[:, 0], y=coords_erased[:, 1], 
            s=100, color="grey", ax=axes[1] # Grey because politics is gone!
        )
        for i, txt in enumerate(topic_labels):
            if i % 5 == 0:
                axes[1].text(coords_erased[i,0]+0.02, coords_erased[i,1], txt, fontsize=9)

        axes[1].set_title("After Linear Erasure (Pure Semantics)")
        axes[1].set_xlabel("Principal Component 1 (Politics Removed)")

        plt.tight_layout()
        plt.show()

        exit()

    dataloaders = get_dataloaders("subreddits", batch_size=32, split=False, renew_cache=False)
    embeddings, partisan_labels, _ = generate_embeddings(dataloaders['train'], path = "data/fine_tuned_sbert")
    temp_ds = Dataset.from_dict({'text': dataloaders['train'].dataset['text']})
    ds = predict_topics(temp_ds, topic_model)

    if(test):
        from collections import Counter
        import pandas as pd
        topic_counts = Counter(ds['topic'])
        print(topic_counts)
        X_all = np.array(embeddings)
        topics_all = np.array(ds['topic'])
        print(f"Embedding shape: {X_all.shape}")
        TOP_N = 6
        top_topic_labels = pd.Series(topics_all).value_counts().nlargest(TOP_N).index.tolist()

        print(f"Filtering for top topics: {top_topic_labels}")

        # Create a boolean mask for just these topics
        mask = np.isin(topics_all, top_topic_labels)

        # Apply mask to get clean arrays
        X_filtered = X_all[mask]
        topics_filtered = topics_all[mask]
        eraser = LeaceEraser.fit(torch.tensor(X_filtered), torch.tensor(topics_filtered))
        X_erased = eraser(torch.tensor(X_filtered)).numpy()
        reducer = umap.UMAP(random_state=42)

        print("Running UMAP on Original...")
        coords_orig = reducer.fit_transform(X_filtered)

        print("Running UMAP on Erased...")
        coords_erased = reducer.fit_transform(X_erased)

        # 5. Plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Helper to make plots pretty
        def plot_umap(ax, coords, labels, title):
            sns.scatterplot(
                x=coords[:, 0], y=coords[:, 1], 
                hue=labels, palette='tab10', s=15, alpha=0.7, ax=ax
            )
            ax.set_title(title, fontsize=14)
            ax.legend(title="Topic", bbox_to_anchor=(1, 1))

        plot_umap(axes[0], coords_orig, topics_filtered, "Original: Topics Split by Ideology?")
        plot_umap(axes[1], coords_erased, topics_filtered, "Erased: Do Topics Merge?")

        plt.tight_layout()
        plt.show()

        exit()
    topics = torch.tensor(ds['topic']).long().to(embeddings.device)

    #old_cpu = old_topics.cpu().numpy() if hasattr(old_topics, 'cpu') else old_topics
    new_cpu = topics.cpu().numpy() if hasattr(topics, 'cpu') else topics

    # Calculate similarity
    """score = adjusted_rand_score(old_cpu, new_cpu)
    print(f"Topic Similarity Score (ARI): {score:.4f}") """

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
        eraser = LeaceEraser.fit(embeddings, partisan_labels)
        embeddings_erased = eraser(embeddings)
        
    print(topic_model.get_topic(0)[:3])
    print(f"Erased embeddings shape: {embeddings_erased.shape}")

    umap_plot(embeddings, partisan_labels, title="UMAP of Original Embeddings (Colored by Partisan Labels)")
    umap_plot(embeddings_erased, partisan_labels, title="UMAP of Embeddings After Erasure (Colored by Partisan Labels)")
    umap_plot(embeddings_erased, topics, title="UMAP of Embeddings After Erasure (Colored by Topic Labels)")


    
    svm_test(
        embeddings,
        partisan_labels
    )

    svm_test(
        embeddings_erased,
        partisan_labels
    )
    run_erasure_two_sources(
        embeddings=embeddings,
        embeddings_erased=embeddings_erased,
        topic_labels=partisan_labels,
        k=20,
        top_k_retrieval=30,
        max_n_clusters=32,
        source_1 = "left",
        source_2 = "right")
   

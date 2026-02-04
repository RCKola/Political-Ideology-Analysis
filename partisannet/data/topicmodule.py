from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from datasets import Dataset
import os
from partisannet.data.datamodule import get_dataloaders
from partisannet.models.get_embeddings import generate_embeddings
from sklearn.cluster import KMeans
import numpy as np

def get_topics(docs: list[str], num_topics: int | None = None, remove_stopwords: bool = False, embeddings=None, embedding_model=None) -> tuple[BERTopic, list[int], list[float]]:

    vectorizer = CountVectorizer(stop_words='english' if remove_stopwords else None, ngram_range=(1,2))
    hdbscan_model = None
    if num_topics is not None:
        print(f"Using K-Means to force {num_topics} balanced clusters...")
        hdbscan_model = KMeans(n_clusters=num_topics, random_state=42)
        # Set nr_topics to None because the clustering model already handles the count
        nr_topics_arg = None 
    else:
        # Fallback to default behavior if no number is given
        nr_topics_arg = "auto"

    topic_model = BERTopic(nr_topics=nr_topics_arg, vectorizer_model=vectorizer, embedding_model=embedding_model, hdbscan_model=hdbscan_model)
    
    if embeddings is not None:
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    else:
        topics, probs = topic_model.fit_transform(docs)


    return topic_model, topics, probs

def include_topics(dataset: Dataset, num_topics: int | None = None, remove_stopwords: bool = True, cluster_in_k: int | None = None, embeddings = None, embedding_model=None) -> tuple[Dataset, object]:
    """Adds topic modeling information to the dataset using BERTopic."""

    docs = [text for text in dataset['text']]
    
    emb_numpy = None
    if embeddings is not None:
        if hasattr(embeddings, "cpu"):
            emb_numpy = embeddings.cpu().numpy()
        else:
            emb_numpy = embeddings

    # 1. Initial Fit (~200 topics)
    topic_model, topics, probs = get_topics(docs, num_topics=num_topics, remove_stopwords=remove_stopwords, embeddings=embeddings, embedding_model=embedding_model)
   

    # 2. Reduce Topics
    if cluster_in_k is not None:
        print(f"Reducing topics to {cluster_in_k}...")
        
        # A. Perform reduction
        topic_model.reduce_topics(docs, nr_topics=cluster_in_k)
        
        # B. Force-fetch the new topics (The Safe Fix)
        # We use transform() instead of .topics_ to ensure we get the latest state
        if emb_numpy is not None:
            topics, _ = topic_model.transform(docs, embeddings=emb_numpy)
        else:
            topics,_ = topic_model.transform(docs)
    

    # 3. Reduce Outliers
    # Now we pass the GUARANTEED reduced list (0..19) to the outlier reducer
    if -1 in topics and emb_numpy is not None:
        print("Reducing outliers...")
        new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=emb_numpy)
        
        # Update the model and our local variable
        topic_model.update_topics(docs, topics=new_topics, vectorizer_model=topic_model.vectorizer_model)
        topics = new_topics
        print(f"DEBUG: Final topic count after outlier reduction: {len(set(topics))}")

    # 4. Update Dataset Columns
    if 'topic' in dataset.column_names:
        dataset = dataset.remove_columns('topic')
    if 'topic_prob' in dataset.column_names:
        dataset = dataset.remove_columns('topic_prob')

    dataset = dataset.add_column('topic', topics)

    return dataset, topic_model


def predict_topics(dataset: Dataset, topic_model, embeddings=None) -> Dataset:
    """Predicts topics on a new dataset using a pre-trained BERTopic model."""
    
    docs = [text for text in dataset['text']]

    # 1. Prepare Embeddings (Tensor -> Numpy)
    emb_numpy = None
    if embeddings is not None:
        if hasattr(embeddings, "cpu"):
            emb_numpy = embeddings.cpu().numpy()
        else:
            emb_numpy = embeddings

    # 2. Predict (Transform)
    # CRITICAL: We use .transform(), not .fit_transform()
    if emb_numpy is not None:
        topics, probs = topic_model.transform(docs, embeddings=emb_numpy)
    else:
        topics, probs = topic_model.transform(docs)

    # 3. Clean up and Add to Dataset
    if 'topic' in dataset.column_names:
        dataset = dataset.remove_columns('topic')
        
    dataset = dataset.add_column('topic', topics)
    # Optional: Add probabilities if you need them
    dataset = dataset.add_column('topic_prob', probs) 

    return dataset

def load_topic_model(dataset_name: str, embedding_model=None, renew_cache=False, num_topics=None, cluster_in_k=20) -> BERTopic:
    """Loads a BERTopic model from disk, optionally with a specified embedding model."""

    cache_dir = os.path.join("data", "cached_data")

    
    topic_model_cache_path = os.path.join(cache_dir, "cached_trained_topic_model" + "_" + dataset_name)

    if os.path.exists(topic_model_cache_path) and not renew_cache:
        print("Loading topic model from disk...")
        topic_model = BERTopic.load(topic_model_cache_path, embedding_model=embedding_model)
        centroid_matrix = np.load(os.path.join(topic_model_cache_path, "centroid_matrix.npy"))
        topic_labels = np.load(os.path.join(topic_model_cache_path, "topic_labels.npy"), allow_pickle=True).tolist()
    else:
        dataloaders = get_dataloaders(dataset_name, batch_size=32, split=False, renew_cache=renew_cache)
        embeddings, partisan_labels, _ = generate_embeddings(dataloaders['train'], path = embedding_model)
        temp_ds = Dataset.from_dict({'text': dataloaders['train'].dataset['text']})
        ds, topic_model = include_topics(temp_ds, remove_stopwords=True, num_topics = num_topics, cluster_in_k= cluster_in_k, embeddings=embeddings, embedding_model=embedding_model)
        topic_model.save(topic_model_cache_path, serialization="safetensors", save_embedding_model=True)
        print("Sample topics from dataset:")
        for i in range(5):
            print(f"Doc: {ds[i]['text'][:50]}... Topic: {ds[i]['topic']}")
        unique_topics = sorted(list(set(ds['topic'])))
        if -1 in unique_topics: unique_topics.remove(-1)

        centroid_matrix = []
        topic_labels = []
        for topic in unique_topics:
            
            topic_embeddings = embeddings[ds['topic'] == topic]
            centroid = topic_embeddings.mean(axis=0)
            centroid_matrix.append(centroid)
            topic_labels.append(topic_model.get_topic_info(topic)['Name']) 
        centroid_matrix =  np.array(centroid_matrix)
        np.save(os.path.join(topic_model_cache_path, "centroid_matrix.npy"), centroid_matrix)
        np.save(os.path.join(topic_model_cache_path, "topic_labels.npy"), np.array(topic_labels, dtype=object))
    return topic_model, centroid_matrix, topic_labels


if __name__ == "__main__":
    print("hello")
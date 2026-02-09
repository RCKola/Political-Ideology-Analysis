from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from datasets import Dataset
import os
from partisannet.data.datamodule import get_dataloaders
from partisannet.models.get_embeddings import generate_embeddings
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
import spacy
import joblib
import tqdm

def lemmatize_docs(docs):
    """Lemmatizes a list of strings using spaCy."""
    print("Pre-lemmatizing documents (this takes a moment)...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    except OSError:
         raise OSError("Please run 'python -m spacy download en_core_web_sm' first.")
    
    clean_docs = []
    # Using nlp.pipe is much faster for processing a list of texts
    for doc in tqdm.tqdm(nlp.pipe(docs, batch_size=1000), total=len(docs)):
        lemmas = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        clean_docs.append(" ".join(lemmas))
    return clean_docs


def get_topics(docs: list[str], num_topics: int | None = None, remove_stopwords: bool = True, embeddings=None, embedding_model=None) -> tuple[BERTopic, list[int], list[float]]:
    clean_docs = lemmatize_docs(docs)   

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words='english' if remove_stopwords else None,
    )
    hdbscan_model = None
    if num_topics is not None:
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=80, # LOW value = granular topics (Default is usually larger)
            metric='euclidean', 
            cluster_selection_method='eom', 
            prediction_data=True
        )
        # Set nr_topics to None because the clustering model already handles the count
        umap_model = UMAP(
            n_neighbors=40,      # LOW value = granular topics
            n_components=5, 
            min_dist=0.0, 
            metric='cosine', 
            random_state=42
        )
        nr_topics_arg = None 
    else:
        # Fallback to default behavior if no number is given
        nr_topics_arg = "auto"

    topic_model = BERTopic(nr_topics=nr_topics_arg, vectorizer_model=vectorizer, embedding_model=embedding_model, hdbscan_model=hdbscan_model, umap_model=umap_model)
    
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
    doc_lengths = [len(d) for d in docs]

    print(f"Max document length: {max(doc_lengths)}")
    print(f"Average document length: {sum(doc_lengths) / len(doc_lengths)}")

    # Find the index of the problematic document
    import numpy as np
    huge_docs_indices = np.where(np.array(doc_lengths) > 1000000)[0]
    print(f"Indices of huge docs: {huge_docs_indices}")

    # Print a preview of the huge doc to see what it is
    if len(huge_docs_indices) > 0:
        idx = huge_docs_indices[0]
        print(f"\nPreview of Doc {idx}: {docs[idx][:200]}...")
        
    emb_numpy = None
    if embeddings is not None:
        if hasattr(embeddings, "cpu"):
            emb_numpy = embeddings.cpu().numpy()
        else:
            emb_numpy = embeddings

    # 1. Initial Fit (~200 topics)
    topic_model, topics, probs = get_topics(docs, num_topics=num_topics, remove_stopwords=remove_stopwords, embeddings=embeddings, embedding_model=embedding_model)
   
    print(f"Initial topic count: {len(set(topics))}")
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
        percentage = np.load(os.path.join(topic_model_cache_path, "topic_percentages.npy"))
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
        clf_tot = joblib.load("data/svm/svm_model.joblib")
        predictions = clf_tot.predict(embeddings)
        centroid_matrix = []
        topic_labels = []
        percentage = []
        for topic in unique_topics:
            topics_array = np.array(ds['topic'])    
            mask = (topics_array == topic)
            topic_embeddings = embeddings[mask]
            avg_topic_predictions = np.mean(predictions[mask])
            print(f"Topic {topic} average prediction (Right-Leaning %): {avg_topic_predictions:.4f}")
            percentage.append(avg_topic_predictions)
            print(f"Topic {topic} raw shape: {topic_embeddings.shape[0]}")
            centroid = topic_embeddings.mean(axis=0)
            print(f"Topic {topic} centroid shape: {centroid.shape}")
            centroid_matrix.append(centroid)
            topic_labels.append(topic_model.get_topic_info(topic)['Name']) 
        centroid_matrix =  np.array(centroid_matrix)
        print(centroid_matrix.shape)
        np.save(os.path.join(topic_model_cache_path, "centroid_matrix.npy"), centroid_matrix)
        np.save(os.path.join(topic_model_cache_path, "topic_labels.npy"), np.array(topic_labels, dtype=object))
        np.save(os.path.join(topic_model_cache_path, "topic_percentages.npy"), np.array(percentage))
    return topic_model, centroid_matrix, topic_labels, percentage


if __name__ == "__main__":
    print("hello")
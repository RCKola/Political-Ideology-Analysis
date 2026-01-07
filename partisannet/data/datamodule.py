from datasets import load_dataset, Dataset
from datasets import ClassLabel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
from datasets import load_from_disk
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path


def combine_text_batched(dataset: Dataset) -> dict[str, list[str]]:
    """Combines 'Title' and 'Text' columns into a single 'text' column.
    Args:
        dataset: A dataset object containing 'Title' and 'Text' columns.
    Returns:
        dict[str, list[str]]: A dictionary with new 'text' column for mapping.
    """
    combined_texts = []
    for title, body in zip(dataset["Title"], dataset["Text"]):
        parts = [title, body]
        clean_row = " ".join([str(p) for p in parts if p])
        combined_texts.append(clean_row)
    return {"text": combined_texts}

def load_datasets(dataset_name: str) -> Dataset:
    """Load dataset based on the given name.
    Args:
        dataset_name (str): Name of the dataset to load. Supported: "mbib-base", "LibCon"
    Returns:
        ds (Dataset): The loaded dataset.
    """
    if dataset_name == "mbib-base":
        ds = load_dataset("mediabiasgroup/mbib-base", split="political_bias")
    elif dataset_name == "LibCon":
        file_path = "file_name.csv"
        ds = kagglehub.dataset_load(
            KaggleDatasetAdapter.HUGGING_FACE,
            "neelgajare/liberals-vs-conservatives-on-reddit-13000-posts",
            file_path,
            pandas_kwargs={"encoding": "latin1", "compression": "zip"}
        )
        
        ds = ds.rename_column("Political Lean", "label")
        new_features = ds.features.copy()
        new_features["label"] = ClassLabel(names=["Liberal", "Conservative"]) # 0: Liberal, 1: Conservative
        ds = ds.cast(new_features)
        ds = ds.map(combine_text_batched, batched=True, num_proc=4)
        num = ds["label"].count(1)
        indices_to_drop = [i for i, label in enumerate(ds["label"]) if label == 0][num:]
        indices_to_keep = [i for i in range(len(ds)) if i not in indices_to_drop]
        ds = ds.select(indices_to_keep)
        ds = ds.select_columns(["self_text", "subreddit"])
    elif dataset_name == "DemRep":
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        csv_path = project_root / "data" / "Training_data" / "training_data.csv"
        ds = load_dataset("csv", data_files=str(csv_path), split="train")
        ds = ds.rename_column("full_text", "text")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return ds

def get_datasets(dataset: str, num_topics = None, cluster_in_k = None) -> dict[str, DataLoader]:
    """Load dataset and return dataloaders for training, validation, and testing.
    Args:
        dataset (str): Name of the dataset to load. Supported: "mbib-base", "LibCon"
        batch_size (int): Batch size for the dataloaders.
    Returns:
        dict[str, DataLoader]: A dictionary containing 'train', 'val', and 'test' dataloaders.
    """
    dst = load_datasets(dataset)
    print(dst)
    ds, topic_model = include_topics(dst, remove_stopwords=True, num_topics = num_topics, cluster_in_k= cluster_in_k)
    # Initialize tokenizer
    model_name='all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/" + model_name)
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

    # Preprocess dataset
    ds = ds.filter(lambda x: x['text'] is not None)
    ds = ds.map(tokenize, batched=True)
    ds.set_format('torch')
    
    return ds, topic_model

def get_dataloaders(dataset: str, batch_size: int, split = True, num_topics = None, cluster_in_k = None, renew_cache = False) -> dict[str, DataLoader]:
    # Inside your main block
    cache_dir = os.path.join("data", "cached_data")
    
    dataset_cache_path = os.path.join(cache_dir, "cached_dataset_hf")
    topic_model_cache_path = os.path.join(cache_dir, "cached_topic_model")

    print(f"Dataset exists: {os.path.exists(dataset_cache_path)}")
    print(f"Topic Model exists: {os.path.exists(topic_model_cache_path)}")
    print(f"Renew Cache is: {renew_cache}")
    # 2. Check if cache exists
    if os.path.exists(dataset_cache_path) and os.path.exists(topic_model_cache_path) and not renew_cache:
        print("Loading dataset from disk...")
        dataset = load_from_disk(dataset_cache_path)
        topic_model = BERTopic.load(topic_model_cache_path)
    else:
        print("Cache not found. Processing data...")
        

        dataset, topic_model = get_datasets(dataset, num_topics=num_topics, cluster_in_k=cluster_in_k)
        
        # Generate Embeddings
        dataset.save_to_disk(dataset_cache_path)
        print("Saving topic model...")
        topic_model.save(topic_model_cache_path, serialization="safetensors", save_embedding_model=True)
    if not split:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return {"train": loader, "val": loader, "test": loader}, topic_model
    
    split_data = dataset.train_test_split(test_size=0.1, seed=42)
    train_val_data = split_data['train'].train_test_split(test_size=0.2, seed=42)  

    train_loader = DataLoader(train_val_data['train'], batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(train_val_data['test'], batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(split_data['test'], batch_size=batch_size, shuffle=False, num_workers=0)
    data_dict = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    return data_dict, topic_model

def get_topics(docs: list[str], num_topics: int | None = None, remove_stopwords: bool = False, embeddings=None) -> tuple[BERTopic, list[int], list[float]]:
    
    vectorizer = CountVectorizer(stop_words='english' if remove_stopwords else None)
    topic_model = BERTopic(nr_topics=num_topics, vectorizer_model=vectorizer)

    # 2. Add logic to use the embeddings if they exist
    if embeddings is not None:
        # BERTopic requires Numpy arrays, so we convert from PyTorch if needed
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        
        # Pass them into fit_transform
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    else:
        # Fallback to default (Standard SBERT) if no embeddings provided
        topics, probs = topic_model.fit_transform(docs)

    return topic_model, topics, probs

def include_topics(dataset: Dataset, num_topics: int | None = None, remove_stopwords: bool = True, cluster_in_k: int | None = None, embeddings = None) -> tuple[Dataset, object]:
    """Adds topic modeling information to the dataset using BERTopic."""

    docs = [text for text in dataset['text']]
    
    # 1. Initial Fit (~200 topics)
    topic_model, topics, probs = get_topics(docs, num_topics=num_topics, remove_stopwords=remove_stopwords, embeddings=embeddings)
    print(f"DEBUG: Initial topic count: {len(set(topics))}")

    # 2. Reduce Topics
    if cluster_in_k is not None:
        print(f"DEBUG: Reducing topics to {cluster_in_k}...")
        
        # A. Perform reduction
        topic_model.reduce_topics(docs, nr_topics=cluster_in_k)
        
        # B. Force-fetch the new topics (The Safe Fix)
        # We use transform() instead of .topics_ to ensure we get the latest state
        if embeddings is not None:
            emb_numpy = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings
            topics, _ = topic_model.transform(docs, embeddings=emb_numpy)
        else:
            topics,_ = topic_model.transform(docs)
        print(f"DEBUG: Topic count after reduction: {len(set(topics))}")

    # 3. Reduce Outliers
    # Now we pass the GUARANTEED reduced list (0..19) to the outlier reducer
    if -1 in topics:
        print("Reducing outliers...")
        new_topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf")
        
        vectorizer = CountVectorizer(stop_words="english") if remove_stopwords else None
        # Update the model and our local variable
        topic_model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer)
        topics = new_topics
        print(f"DEBUG: Final topic count after outlier reduction: {len(set(topics))}")

    # 4. Update Dataset Columns
    if 'topic' in dataset.column_names:
        dataset = dataset.remove_columns('topic')
    if 'topic_prob' in dataset.column_names:
        dataset = dataset.remove_columns('topic_prob')

    dataset = dataset.add_column('topic', topics)
    dataset = dataset.add_column('topic_prob', probs)

    return dataset, topic_model


if __name__ == "__main__":
    dataloaders, topic_model = get_dataloaders("DemRep", batch_size=128, split=True, num_topics=None, cluster_in_k=40, renew_cache=True)



    
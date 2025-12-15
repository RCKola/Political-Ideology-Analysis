from datasets import load_dataset, Dataset
from datasets import ClassLabel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import kagglehub
from kagglehub import KaggleDatasetAdapter

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
        new_features["label"] = ClassLabel(names=["Conservative", "Liberal"]) # 0: Conservative, 1: Liberal
        ds = ds.cast(new_features)
        ds = ds.map(combine_text_batched, batched=True, num_proc=4)
        ds = ds.remove_columns(["Title", "Text", "Score", "URL","Num of Comments", "Subreddit", "Date Created"])
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return ds

def get_dataloaders(dataset: str, batch_size: int, split = True, num_topics = None) -> dict[str, DataLoader]:
    """Load dataset and return dataloaders for training, validation, and testing.
    Args:
        dataset (str): Name of the dataset to load. Supported: "mbib-base", "LibCon"
        batch_size (int): Batch size for the dataloaders.
    Returns:
        dict[str, DataLoader]: A dictionary containing 'train', 'val', and 'test' dataloaders.
    """
    dst = load_datasets(dataset)
    ds, topic_model = include_topics(dst, remove_stopwords=True, num_topics = num_topics)
    # Initialize tokenizer
    model_name='all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/" + model_name)
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

    # Preprocess dataset
    ds = ds.filter(lambda x: x['text'] is not None)
    ds = ds.map(tokenize, batched=True)
    ds.set_format('torch')
    if not split:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        return {"train": loader, "val": loader, "test": loader}, topic_model
    split_data = ds.train_test_split(test_size=0.1, seed=42)
    train_val_data = split_data['train'].train_test_split(test_size=0.2, seed=42)

    # TODO: revisit dataset splitting strategy
    train_loader = DataLoader(train_val_data['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_val_data['test'], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(split_data['test'], batch_size=batch_size, shuffle=False)
    data_dict = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    return data_dict, topic_model

def include_topics(dataset: Dataset, num_topics: int | None = None, remove_stopwords: bool = True) -> tuple[Dataset, object]:
    """Adds topic modeling information to the dataset using BERTopic.
    Args:
        dataset (Dataset): The dataset to augment with topic information.
        num_topics (int | None): Number of topics to extract. If None, let BERTopic decide.
        remove_stopwords (bool): Whether to remove stopwords during topic modeling.
    Returns:
        dataset, topic_model (tuple[Dataset, object]): The augmented dataset with 'topic' and 'topic_prob' columns, and the trained BERTopic model.
    """
    from partisannet.models.topics import get_topics

    docs = [text for text in dataset['text']]
    topic_model, topics, probs = get_topics(docs, num_topics=num_topics, remove_stopwords=remove_stopwords)

    dataset = dataset.add_column('topic', topics)
    dataset = dataset.add_column('topic_prob', probs)
    return dataset, topic_model

if __name__ == "__main__":
    dataloaders = get_dataloaders("LibCon", batch_size=32)
    for batch in dataloaders['train']:
        print(batch)
        break

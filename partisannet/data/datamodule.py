from datasets import load_dataset, Dataset
from datasets import ClassLabel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
from datasets import load_from_disk
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
    elif dataset_name == "testdata":
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        csv_path = project_root / "data" / "Training_data" / "test_data.csv"
        ds = load_dataset("csv", data_files=str(csv_path), split="train")
    elif dataset_name == "subreddits":
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        csv_path = project_root / "data" / "Training_data" / "subreddits.csv"
        ds = load_dataset("csv", data_files=str(csv_path), split="train")
    elif dataset_name == "topic_data":
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        csv_path = project_root / "data" / "Training_data" / "topics_data.csv"
        ds = load_dataset("csv", data_files=str(csv_path), split="train") 
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return ds

def get_datasets(dataset: str,) -> dict[str, DataLoader]:
    """Load dataset and return dataloaders for training, validation, and testing.
    Args:
        dataset (str): Name of the dataset to load. Supported: "mbib-base", "LibCon"
        batch_size (int): Batch size for the dataloaders.
    Returns:
        dict[str, DataLoader]: A dictionary containing 'train', 'val', and 'test' dataloaders.
    """
    ds = load_datasets(dataset)
    
    # Initialize tokenizer
    model_name='all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/" + model_name)
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

    # Preprocess dataset
    ds = ds.filter(lambda x: x['text'] is not None)
    ds = ds.map(tokenize, batched=True)
    ds.set_format('torch')
    
    return ds

def get_dataloaders(dataset: str, batch_size: int, split = True, renew_cache = False) -> dict[str, DataLoader]:
    cache_dir = os.path.join("data", "cached_data")

    dataset_cache_path = os.path.join(cache_dir, dataset)

    print(f"Dataset exists: {os.path.exists(dataset_cache_path)}")
    
    print(f"Renew Cache is: {renew_cache}")

    if os.path.exists(dataset_cache_path) and not renew_cache:
        print("Loading dataset from disk...")
        dataset = load_from_disk(dataset_cache_path)
    else:
        print("Cache not found. Processing data...")
        dataset = get_datasets(dataset)

        dataset.save_to_disk(dataset_cache_path)

    if not split:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return {"train": loader, "val": loader, "test": loader}
    
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
    return data_dict




if __name__ == "__main__":
    dataloaders, topic_model = get_dataloaders("DemRep", batch_size=128, split=True, num_topics=None, cluster_in_k=40, renew_cache=True)



    
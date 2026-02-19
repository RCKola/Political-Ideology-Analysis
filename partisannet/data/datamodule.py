from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from datasets import load_from_disk
from pathlib import Path



def load_datasets(dataset_name: str) -> Dataset:
    """Load dataset based on the given name.
    Args:
        dataset_name (str): Name of the dataset to load. Supported: "mbib-base", "LibCon"
    Returns:
        ds (Dataset): The loaded dataset.
    """
    
    if dataset_name == "DemRep":
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
        dataset (str): Name of the dataset to load. Supported: "DemRep", "testdata", "subreddits", "topic_data"
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



    
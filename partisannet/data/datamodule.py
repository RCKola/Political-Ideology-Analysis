from datasets import load_dataset
from datasets import ClassLabel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import kagglehub
from kagglehub import KaggleDatasetAdapter

def combine_text_batched(data):
    # 'examples' is a dictionary of lists: {"Title": ["t1", "t2"], "Post Text": ["b1", "b2"]}
    
    combined_texts = []
    
    # We must zip the columns to process them row-by-row
    for title, body in zip(data["Title"], data["Text"]):
        
        # 1. Collect parts
        parts = [title, body]
        
        # 2. Filter & Join
        # - str(p): Converts numbers/non-strings to text safely
        # - if p: Skips None and empty strings ""
        clean_row = " ".join([str(p) for p in parts if p])
        
        combined_texts.append(clean_row)
    
    # Return a dictionary with the new column list
    return {"text": combined_texts}

def get_dataloaders(dataset: str, batch_size: int) -> DataLoader:
    if dataset == "mbib-base":
        
        ds = load_dataset("mediabiasgroup/mbib-base", split="political_bias")
    elif dataset == "LibCon":
        file_path = "file_name.csv"
        ds = kagglehub.dataset_load(
        KaggleDatasetAdapter.HUGGING_FACE,
        "neelgajare/liberals-vs-conservatives-on-reddit-13000-posts",
        file_path,
        pandas_kwargs={"encoding": "latin1",
                       "compression": "zip"
                       }
        # Provide any additional arguments like 
        # sql_query or pandas_kwargs. See the 
        # documenation for more information:
        # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
        )
        
        ds = ds.rename_column("Political Lean", "label")
        new_features = ds.features.copy()
        new_features["label"] = ClassLabel(names=["Conservative", "Liberal"])
        ds = ds.cast(new_features)
        ds = ds.map(combine_text_batched, batched=True)
        ds = ds.remove_columns(["Title", "Text", "Score", "URL","Num of Comments", "Subreddit", "Date Created"])
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    # Initialize tokenizer
    
    model_name='all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/" + model_name)
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

    
    ds = ds.filter(lambda x: x['text'] is not None)
    ds = ds.map(tokenize, batched=True)
    ds.set_format('torch')

    split_data = ds.train_test_split(test_size=0.1, seed=42)
    train_val_data = split_data['train'].train_test_split(test_size=0.2, seed=42)

    train_loader = DataLoader(train_val_data['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_val_data['test'], shuffle=False)
    test_loader = DataLoader(split_data['test'], shuffle=False)
    data_dict = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    return data_dict 

if __name__ == "__main__":
    dataloaders = get_dataloaders("LibCon", batch_size=32)
    for batch in dataloaders['train']:
        print(batch)
        break

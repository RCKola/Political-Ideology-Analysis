from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def get_dataloaders(dataset: str, batch_size: int) -> DataLoader:
    if dataset != "mbib-base":
        raise ValueError(f"Dataset {dataset} not supported.")
    ds = load_dataset("mediabiasgroup/mbib-base", split="political_bias")

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
    dataloaders = get_dataloaders("mbib-base", batch_size=32)
    for batch in dataloaders['train']:
        print(batch)
        break

import torch
import pytorch_lightning as L
from partisannet.data.datamodule import get_dataloaders, get_datasets
from partisannet.modelling import PartisanNetModel
from partisannet.models.classifier import SBERTClassifier
from partisannet.models.get_embeddings import get_finetuned_embeddings
from partisannet.eval.metrics import ClusterMetrics
from train import run_training, get_parser

MODELS_LIST = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v2",
]

def eval_models_clf():
    parser = get_parser()
    args = parser.parse_args([])

    model_paths = []
    results_list = []
    for model_name in MODELS_LIST:
        print(f"Training model: {model_name}")
        args.model_name = model_name
        path, results = run_training(args)
        model_paths.append(path)
    
    return model_paths

def setup():
    dataloaders, _ = get_dataloaders(
        "DemRep", 
        batch_size=256, 
        split=True, 
        num_topics=None
    )
    trainer = L.Trainer(
        accelerator="gpu", 
        devices=1,
    )
    sbert_model = SBERTClassifier()
    model = PartisanNetModel(sbert_model)

    return trainer, model, dataloaders['test']

def encode_text_labels(texts):
    unique_labels = sorted(list(set(texts)))
    mapping = {label: idx for idx, label in enumerate(unique_labels)}

    encoded_labels = [mapping[text] for text in texts]
    return encoded_labels, mapping

def eval_clustering_subreddits(sbert_path = "data/fine_tuned_sbert"):
    dataset = get_datasets("subreddits").shuffle(seed=42)
    embeddings, data_dict = get_finetuned_embeddings(dataset, model_path=sbert_path)
    
    data_dict["embeddings"] = embeddings
    subreddits = dataset["subreddit"]
    encoded_labels, label_mapping = encode_text_labels(subreddits)
    data_dict["subreddit"] = torch.tensor(encoded_labels)

    meter = ClusterMetrics()
    meter.collect(data_dict)
    metrics = meter.compute("embeddings", "subreddit")
    print("Clustering Metrics:", metrics)

def eval_clustering(sbert_path = "data/centerloss_sbert"):
    dataset = get_datasets("testdata").shuffle(seed=42)
    embeddings, data_dict = get_finetuned_embeddings(dataset, model_path=sbert_path)
    data_dict["embeddings"] = embeddings

    meter = ClusterMetrics()
    meter.collect(data_dict)
    metrics = meter.compute("embeddings", "label")
    print("Clustering Metrics:", metrics)

def main():
    # eval_clustering()
    eval_clustering_subreddits()

if __name__ == "__main__":
    main()
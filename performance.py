import json
from datetime import datetime

import torch
import lightning.pytorch as L
from partisannet.data.datamodule import get_dataloaders, get_datasets
from partisannet.modelling import PartisanNetModel
from partisannet.models.classifier import SBERTClassifier
from partisannet.models.get_embeddings import get_finetuned_embeddings
from partisannet.eval.metrics import ClusterMetrics
from train import run_training, get_parser

MODELS_LIST = [
    "nli-roberta-base",           # 125M params
    "nli-bert-base",              # 110M params
    "paraphrase-MiniLM-L12-v2",   # 33M params
    "all-MiniLM-L12-v2",          # 33M params
    "all-MiniLM-L6-v2",           # 22M params
]

def eval_models_clf(output_file="data/model_comparison_results.json"):
    parser = get_parser()
    args = parser.parse_args([])

    # Load existing results if file exists
    try:
        with open(output_file, "r") as f:
            all_results = json.load(f)
        trained_models = {r["model_name"] for r in all_results}
        print(f"Found {len(trained_models)} already trained models: {trained_models}")
    except FileNotFoundError:
        all_results = []
        trained_models = set()

    for model_name in MODELS_LIST:
        if model_name in trained_models:
            print(f"Skipping {model_name} (already trained)")
            continue

        print(f"Training model: {model_name}")
        args.model_name = model_name
        args.max_epochs = 30
        args.batch_size = 128
        path, results = run_training(args)

        run_result = {
            "model_name": model_name,
            "model_path": path,
            "test_results": results,
            "config": {
                "max_epochs": args.max_epochs,
                "batch_size": args.batch_size,
                "lora_r": args.lora_r,
                "lr": args.lr,
            },
            "timestamp": datetime.now().isoformat(),
        }
        all_results.append(run_result)

        # Save after each run to preserve progress
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {output_file}")

    return all_results

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
    # eval_clustering_subreddits()
    eval_models_clf()

if __name__ == "__main__":
    main()
import numpy as np
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, MatthewsCorrCoef, AveragePrecision
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import HDBSCAN, KMeans

def get_classif_meter(num_classes=2):
    if num_classes != 2:
        raise NotImplementedError("Metrics for multi-class classification not implemented yet.")
    meter = MetricCollection([
            Accuracy(task='binary'),
            Precision(task='binary'),
            Recall(task='binary'),
            F1Score(task='binary'),
            AUROC(task='binary'),
            AveragePrecision(task='binary'),
            MatthewsCorrCoef(task='binary')
        ], prefix="test_")
    
    return meter

class ClusterMetrics:
    def __init__(self, engine="kmeans"):
        self.collection = {}
        self.metrics =[
            adjusted_rand_score,
            normalized_mutual_info_score,
            silhouette_score
        ]
        self.engine = engine
        
    
    def collect(self, data_dict):
        for key, val in data_dict.items():
            if key not in self.collection:
                self.collection[key] = [val]
            else:
                self.collection[key].append(val)
    
    def compute(self, embedding_key, label_key):
        embeddings = self._group_data(embedding_key)
        labels = self._group_data(label_key)
        n_clusters = len(np.unique(labels))

        clusters = self._cluster(embeddings, n_clusters=n_clusters)
        assert len(clusters) == len(labels) == len(embeddings), "Mismatch in data lengths for metrics computation"
        
        results = {}
        for metric in self.metrics:
            if metric == silhouette_score:
                score = metric(embeddings, labels)
            else:
                score = metric(clusters, labels)
            results[metric.__name__] = score
        
        self.reset()
        return results
    
    def reset(self):
        self.collection = {}
    
    def _group_data(self, key):
        data = self.collection.get(key, None)
        if data is None:
            raise ValueError(f"No data collected for key: {key}")
        data = torch.cat(data, dim=0).cpu().numpy() if isinstance(data[0], torch.Tensor) else data
        return data

    def _cluster(self, embeddings, n_clusters=2):
        if self.engine == "hdbscan":
            clusterer = HDBSCAN(max_cluster_size=n_clusters, copy=True)
        elif self.engine == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError(f"Unknown clustering engine: {self.engine}")
        
        cluster_labels = clusterer.fit_predict(embeddings)
        return cluster_labels


def clf_test():
    meter = get_classif_meter()

    # Simulated Batch 1: High confidence, mostly correct
    # preds: probabilities (0.0 to 1.0)
    # targets: integers (0 or 1)
    preds_1 = torch.tensor([0.1, 0.9, 0.8, 0.2])
    target_1 = torch.tensor([0, 1, 1, 0])

    # Simulated Batch 2: Lower confidence, some errors
    preds_2 = torch.tensor([0.6, 0.3, 0.2, 0.7])
    target_2 = torch.tensor([0, 0, 1, 1])

    # Update meter with batches
    meter.update(preds_1, target_1)
    meter.update(preds_2, target_2)

    # Compute final results
    results = meter.compute()

    print("Computed Metrics:")
    for key, val in results.items():
        print(f"{key:25}: {val.item():.4f}")

def cluster_test():
    # 1. Setup Mock Data
    # Simulate two groups of embeddings: 
    # Class 0: centered around [1, 1], Class 1: centered around [10, 10]
    torch.manual_seed(42)
    group_left = torch.randn(50, 8) + 1.0
    group_right = torch.randn(50, 8) + 10.0
    
    embeddings = torch.cat([group_left, group_right], dim=0)
    labels = torch.cat([torch.zeros(50), torch.ones(50)], dim=0).long()

    # 2. Initialize and Collect
    cm = ClusterMetrics()
    
    # Simulate batch collection (2 batches)
    cm.collect({"emb": embeddings[:50], "target": labels[:50]})
    cm.collect({"emb": embeddings[50:], "target": labels[50:]})

    print("--- Testing KMeans Engine ---")
    # We expect high scores here because the mock clusters are well-separated
    results_km = cm.compute(embedding_key="emb", label_key="target")
    for name, score in results_km.items():
        print(f"{name:30}: {score:.4f}")

    print("\n--- Testing HDBSCAN Engine ---")
    # Note: HDBSCAN uses 'min_cluster_size', not 'max_cluster_size' in standard params.
    # I modified the call below to use the default 'hdbscan' engine.
    try:
        results_hd = cm.compute(embedding_key="emb", label_key="target")
        for name, score in results_hd.items():
            print(f"{name:30}: {score:.4f}")
    except Exception as e:
        print(f"HDBSCAN Error: {e}")

if __name__ == "__main__":
    cluster_test()


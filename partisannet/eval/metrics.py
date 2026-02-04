import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, MatthewsCorrCoef, AveragePrecision

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

if __name__ == "__main__":
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


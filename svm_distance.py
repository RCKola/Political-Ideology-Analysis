import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns


import umap
import joblib

# 1. Setup Imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path: sys.path.append(project_root)

from partisannet.data.datamodule import get_dataloaders
from partisannet.models.get_embeddings import generate_embeddings

def main():
    # --- CONFIG ---
    # --------------

    # 1. Load Data
    print("Loading Data...")
    dataloaders = get_dataloaders("testdata", batch_size=32, split=False, renew_cache=False)
    
    
    
    embeddings, labels,_ = generate_embeddings(dataloaders['train'], path = "data/centerloss_sbert")
    # 3. Train Linear SVM (To find the decision boundary)
    print("Training Linear SVM to find the 'Line'...")
    svm = joblib.load("data/svm/svm_model.joblib")
    # 4. Calculate Distances
    # decision_function returns the signed distance to the hyperplane
    # Positive = Class 1 (Right), Negative = Class 0 (Left)
    dists = svm.decision_function(embeddings)
    
    # --- PLOT 1: HISTOGRAM (Separability) ---
    print("Plotting Histogram...")
    plt.figure(figsize=(10, 6))
    
    # Plot Class 0 (Left)
    sns.histplot(dists[labels==0], color='blue', label='Democrat (Class 0)', kde=True, alpha=0.6)
    # Plot Class 1 (Right)
    sns.histplot(dists[labels==1], color='red', label='Republican (Class 1)', kde=True, alpha=0.6)
    
    plt.axvline(0, color='black', linestyle='--', label='Decision Boundary (Neutral)')
    plt.title("Distribution of SVM Distances (Ideological Polarization)")
    plt.xlabel("Distance from Hyperplane (Left < 0 < Right)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("Results/svm_distance_histogram.png")
    plt.show()

    # --- PLOT 2: UMAP COLORED BY DISTANCE ---
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    print("Plotting UMAP gradient...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        c=dists,           # COLOR BY DISTANCE
        cmap='coolwarm',   # Blue -> White -> Red
        s=10, 
        alpha=0.8,
        vmin=-2, vmax=2    # Clamp extreme values for better contrast
    )
    plt.colorbar(scatter, label="SVM Distance (Blue=Left, Red=Right)")
    plt.title("UMAP Colored by SVM Distance (The Linear Direction)")
    plt.savefig("Results/svm_distance_umap.png")
    plt.show()

if __name__ == "__main__":
    main()
import sys
import os
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# 1. Setup Imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path: sys.path.append(project_root)

from partisannet.data.datamodule import get_dataloaders
from partisannet.models.get_embeddings import generate_embeddings

def main():
    # --- CONFIG ---
    MODEL_PATH = "data/fine_tuned_sbert"  # Your saved SBERT folder
    SAMPLE_SIZE = 10000                    # Number of points to plot
    # --------------

    # 1. Load the Original Binary Data
    print("Loading LibCon Test Data...")
    # split=True gives us train/val/test. We use 'test'.
    dataloaders, topic_model = get_dataloaders("DemRep", batch_size=32, split=False, num_topics=None, cluster_in_k=40, renew_cache=False)
    
    
    
    embeddings, labels, old_topics = generate_embeddings(dataloaders['train'], path = "data/fine_tuned_sbert")

    # 4. Run UMAP Projection
    print("Running UMAP (384d -> 2d)...")
    # n_neighbors=15 and min_dist=0.1 are standard defaults for structure
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    # 5. Plot
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot
    # We use a discrete colormap (CoolWarm) where 0=Blue, 1=Red
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        c=labels, 
        cmap='coolwarm', 
        s=10, 
        alpha=0.7
    )
    
    # Custom Legend
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Left (0)', 'Right (1)'])
    
    plt.title("UMAP of Binary Partisan Embeddings (Fine-Tuned)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    output_file = "umap_binary.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
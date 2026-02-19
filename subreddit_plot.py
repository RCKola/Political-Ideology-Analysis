
from partisannet.data.datamodule import get_dataloaders
from partisannet.models.get_embeddings import generate_embeddings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
from sklearn.decomposition import PCA

if __name__ == "__main__":
    
    linerasure = True
    two_axis = False
    dataloaders = get_dataloaders("subreddits", batch_size=32, split=False, renew_cache=False)
    embeddings, partisan_labels, subreddits = generate_embeddings(dataloaders['train'], path = "data/centerloss_sbert_full")
    
    cf_svm = joblib.load("data/svm/svm_model.joblib")
    labels = cf_svm.predict(embeddings)
    
    df =  pd.DataFrame({"embedding": list(embeddings), "subreddit": subreddits, "label": labels})
    


    
    subreddit_centroids = df.groupby('subreddit')['embedding'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).to_dict()
    avg_labels = df.groupby('subreddit')['label'].apply(
        lambda x: np.mean(x)
    ).to_dict()

    print(subreddit_centroids.keys())
   
    eraser = joblib.load("data/svm/linear_eraser.joblib")
    X_centroids = np.vstack(list(subreddit_centroids.values()))
    X_concept = X_centroids - eraser(torch.tensor(X_centroids)).numpy() 
    
    pca_1d = PCA(n_components=1)
    concept_scalar = -1 * pca_1d.fit_transform(X_concept).flatten()

    df_plot = pd.DataFrame({
        "Subreddit": list(subreddit_centroids.keys()),
        "Political_Score": concept_scalar,
        "% Right": list(avg_labels.values())  #0 (Left) to 1 (Right)
    })

    # 3. Sort by the score to see the spectrum (Left <-> Right)
    df_plot = df_plot.sort_values("Political_Score")

    # --- VISUALIZATION ---
    
    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(6, 10))

    # Plot A: The Spectrum (Sorted Bar Chart)
    # This shows exactly where each subreddit falls on the "Politics Line"
    sns.barplot(
        data=df_plot, 
        y="Subreddit", 
        x="Political_Score", 
        hue="% Right", 
        palette="coolwarm", # Blue (Left) to Red (Right) usually
        dodge=False,        # Makes bars align with x-axis ticks
        orient = "h"
    )
    plt.title("") 
    plt.ylabel("") 
    plt.xlabel("Magnitude of Partisan Vector")
    plt.legend(title="% Right-Leaning", loc="upper right")    

    plt.tight_layout()
    plt.savefig("Results/subreddit_bias_plot.pdf", format="pdf", bbox_inches='tight')
    plt.show()
    
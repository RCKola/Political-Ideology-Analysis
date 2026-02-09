
from concept_erasure import LeaceEraser
from partisannet.data.topicmodule import load_topic_model
from partisannet.data.datamodule import get_dataloaders
from partisannet.models.get_embeddings import generate_embeddings
import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from numpy import dot
import seaborn as sns
from adjustText import adjust_text
import joblib
import torch

if __name__ == "__main__":
    
    linerasure = True
    
    dataloaders = get_dataloaders("subreddits", batch_size=32, split=False, renew_cache=False)
    embeddings, partisan_labels, subreddits = generate_embeddings(dataloaders['train'], path = "data/centerloss_sbert")
    df =  pd.DataFrame({"embedding": list(embeddings), "subreddit": subreddits})
    
    subreddit_centroids = df.groupby('subreddit')['embedding'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).to_dict()
    print(subreddit_centroids.keys())
    # 1. Define the poles
     # The "Right" Pole

    # 2. Construct the Axis Vector
    # (Dem - Con) creates a vector pointing towards the Democrat side
    if linerasure:
        X_centroids = np.vstack(list(subreddit_centroids.values()))
        svm = joblib.load("data/svm/svm_model.joblib")
        labels = svm.predict(X_centroids)
        eraser = LeaceEraser.fit(torch.tensor(X_centroids), torch.tensor(labels))

        X_erased = eraser(torch.tensor(X_centroids)).numpy()
        from sklearn.decomposition import PCA

        # Fit PCA on the ORIGINAL to define the axes, then transform both
        # This keeps the "view" consistent so you can see them move
        pca = PCA(n_components=2)
        coords_orig = pca.fit_transform(X_centroids)
        coords_erased = pca.transform(X_erased) # Use same axes!

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot A: Original Centroids
        # Color by their position on the political axis (X-axis usually captures variance)
        sns.scatterplot(
            x=coords_orig[:, 0], y=coords_orig[:, 1], 
            s=100, hue=labels, palette="coolwarm", legend=True, ax=axes[0]
        )
        # Label a few important topics
    

        axes[0].set_title("Original Topic Centroids (Politicized)")
        axes[0].set_xlabel("Principal Component 1 (Likely Political)")

        # Plot B: Erased Centroids
        sns.scatterplot(
            x=coords_erased[:, 0], y=coords_erased[:, 1], 
            s=100, hue=coords_orig[:, 0], palette="coolwarm", legend=False, ax=axes[1] # Grey because politics is gone!
        )

        axes[1].set_title("After Linear Erasure (Pure Semantics)")
        axes[1].set_xlabel("Principal Component 1 (Politics Removed)")

        plt.tight_layout()
        plt.show()

        exit()
    political_axis = np.load("data/cached_data/political_axis.npy")
    scores = {}
    for sub, centroid in subreddit_centroids.items():
        # Project the centroid onto the axis
        # We normalize the axis so the scale is interpretable
        score = dot(centroid, political_axis) / norm(political_axis)
        scores[sub] = score

    # Convert to DataFrame for easy viewing
    

    # ... (Your previous code for loading data and calculating 'scores' goes here) ...

    # Assuming 'scores' is your dictionary of {subreddit: score}
    # and 'subreddit_centroids' is available.

    # Convert to DataFrame for easy viewing
    results = pd.DataFrame(list(scores.items()), columns=['Subreddit', 'Political_Score'])
    results = results.sort_values('Political_Score')

    # 2. Increase figure size for better readability
    plt.figure(figsize=(14, 6))

    # 3. REVERSE PALETTE to 'coolwarm_r'
    # Red = Conservative (negative), Blue = Democratic (positive)
    sns.stripplot(data=results, x='Political_Score', y=[''] * len(results),
                jitter=False, s=15, hue='Political_Score', palette='coolwarm_r', legend=False)

    # 4. Collect text objects for adjustment
    texts = []
    for i, row in results.iterrows():
        # Place text slightly above the point initially
        texts.append(plt.text(row.Political_Score, 0.02, row.Subreddit, fontsize=10))

    # 5. Automatically adjust text positions to avoid overlap
    adjust_text(texts,
                # Push labels away from each other and from the points
                force_points=0.2, force_text=0.5,
                expand_points=(1, 1), expand_text=(1, 1),
                # Add connecting lines
                arrowprops=dict(arrowstyle='-', color='grey', alpha=0.5, lw=1)
                )

    # Formatting
    plt.title("Subreddit Bias Projection (SBERT Embeddings)", fontsize=14)
    plt.xlabel("← More Conservative (Red)  |  More Democratic (Blue) →", fontsize=12)
    plt.yticks([]) # Hide y-axis
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5) # Center line

    # Remove the border for a cleaner look
    sns.despine(left=True, bottom=True)
    plt.savefig("Plots/subreddit_bias_plot.png", bbox_inches='tight')
    print("Plot saved to Plots/subreddit_bias_plot.png")
    plt.show()
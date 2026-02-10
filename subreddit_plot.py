
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
    
    linerasure = False
    two_axis = True
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

    political_axis = -1*np.load("data/cached_data/political_axis.npy")
    axis_norm = political_axis / np.linalg.norm(political_axis)
    scores = {}
    for sub, centroid in subreddit_centroids.items():
        # Project the centroid onto the axis
        # We normalize the axis so the scale is interpretable
        score = np.dot(centroid, axis_norm)
        scores[sub] = score


    if two_axis:
        plot_data = []
        for sub in scores.keys():
            plot_data.append({
                'Subreddit': sub,
                'Geometric_Score': scores[sub],          # X-Axis
                'Pct_Republican': avg_labels[sub] * 100  # Y-Axis (Converted to %)
            })

        df_plot = pd.DataFrame(plot_data)

        # --- 4. Generate the Plot ---
        plt.figure(figsize=(14, 10))

# Set a clean background style
        sns.set_style("whitegrid")

        # FIX 2: Add Color Scale mapped to Y-axis value
        # palette='vlag' diverges from Blue (low values) to Red (high values)
        sns.scatterplot(
            data=df_plot,
            x='Geometric_Score',
            y='Pct_Republican',
            hue='Pct_Republican',  # Map color to percentage
            palette='vlag',        # Red-Blue diverging palette
            s=250,                 # Large, visible dots
            edgecolor='black',     # distinct borders
            linewidth=1,
            alpha=0.9,
            legend=False           # Hide legend (color explains itself based on Y-axis)
        )

        # Add center reference lines
        plt.axvline(0, color='#555555', linestyle='--', alpha=0.5)
        plt.axhline(50, color='#555555', linestyle='--', alpha=0.5)

        # FIX 3: Make Labels Readable with adjustText
        texts = []
        for i in range(df_plot.shape[0]):
            # We create text objects but don't add them to the plot immediately
            texts.append(plt.text(
                df_plot.Geometric_Score[i],
                df_plot.Pct_Republican[i],
                df_plot.Subreddit[i],
                fontsize=11,
                fontweight='bold',
                color='#333333' # Dark grey text is softer than pure black
            ))

        # This function iteratively moves the text objects to prevent overlap
        # arrowprops adds nice little lines connecting text to the dot if it moved far
        adjust_text(texts,
                    arrowprops=dict(arrowstyle='-', color='grey', alpha=0.6),
                    expand_points=(1.5, 1.5) # Push text slightly further from dots
                )

        # Final Formatting with corrected X-axis label
        plt.title("Subreddit Alignment: Geometric Bias vs. Classifier Prediction", fontsize=18, pad=20)
        plt.xlabel("Geometric Partisan Score (Projected on Axis)\n← Liberal  |  Conservative →", fontsize=14, labelpad=10)
        plt.ylabel("Percentage of Texts Classified as Republican (%)", fontsize=14, labelpad=10)

        # Set axis limits to make it look clean (adjust based on your actual data range)
        plt.xlim(-2.2, 2.2)
        plt.ylim(-5, 105)

        plt.tight_layout()
        plt.show()
        exit()
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
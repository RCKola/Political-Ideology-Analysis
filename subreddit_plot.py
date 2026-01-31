
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

if __name__ == "__main__":
    

    # Load pre-trained topic model and data
    trained_embeddings = True

    
    dataloaders = get_dataloaders("subreddits", batch_size=32, split=False, renew_cache=False)
    embeddings, partisan_labels, subreddits = generate_embeddings(dataloaders['train'], path = "data/fine_tuned_sbert")
    df =  pd.DataFrame({"embedding": list(embeddings), "subreddit": subreddits})
    
    subreddit_centroids = df.groupby('subreddit')['embedding'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).to_dict()
    print(subreddit_centroids.keys())
    # 1. Define the poles
    vec_dem = subreddit_centroids['democrats']       # The "Left" Pole
    vec_con = subreddit_centroids['Conservative']   # The "Right" Pole

    # 2. Construct the Axis Vector
    # (Dem - Con) creates a vector pointing towards the Democrat side
    political_axis = vec_dem - vec_con

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
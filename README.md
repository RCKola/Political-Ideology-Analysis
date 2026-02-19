# PartisanNet: Untangling Political Ideology Representations in Online Communities

**Authors:** Thibault Cangemi & Zhiang Chen  
**Institution:** ETH Zürich  
**Course:** Supervised Research (Law, Economics, and Data Science)

## Overview

This repository serves as the official replication package for our paper, *"PartisanNet: Untangling Political Ideology Representations in Online Communities."* It contains all necessary code, data subsets, and instructions to reproduce the contrastive fine-tuning, LEACE concept erasure, and clustering results discussed in the text.

## 1. Repository Structure

```text
Political-Ideology-Analysis/
├── data/                    # Contains the csv files of the subsets of PartisanNet Dataset and all the cached models
├── partisannet/             
│   ├── data/                # Code for loading dataloaders and topic models
│   └── models/              # The core SentenceTransformer-based model used in this paper
├── results/                 # Output folder for generated plots and tables
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 2. Environment Setup

To ensure reproducibility, please create a fresh virtual environment and install the required dependencies.

```bash
# Clone the repository
git clone [https://github.com/RCKola/Political-Ideology-Analysis](https://github.com/RCKola/Political-Ideology-Analysis)
cd PartisanNet

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

*(Key dependencies include `torch`, `sentence-transformers`, `bertopic`, `concept-erasure`, and `scikit-learn`.)*

## 3. Data Access

Because the full datasets are too large to host on GitHub, we have split the data access into two parts:

* **Experiment Subset (Ready to Run):** The exact subset of 50,000 comments across 30 subreddits used for our primary experiments (Sections 5.2, 5.3, and 5.4) is provided in `data/Training_data/*`. **You can run the replication code immediately using this file.**
* **Full Datasets (Optional):** The full "Public Opinion on Republicans (Rep-Ops)" and "Public Opinion on Democrats (Dem-Ops)" datasets can be downloaded from Kaggle (Asaniczka, 2024a; Asaniczka, 2024b).

## 4. Replication Guide

To reproduce the findings, tables, and figures from the paper, run the files in the following order:

### `train.py`

* **Purpose:** Demonstrates the contrastive fine-tuning process of the `all-MiniLM-L6-v2` backbone.
* **Note:** Full training takes approximately 50 epochs on an RTX 4060ti. For convenience, we have provided the pre-trained weights in the `data/centerloss_sbert_full/` directory so you can skip training and proceed directly to the analysis.

### `Augment_Model.py`

* **Purpose:** Sets up the centerloss_sbert model needed for extracting the embeddings.
* **Output:** Saves an the trained model in data/centerloss_sbert_full

### `svm_train.py`

* **Purpose:** Trains the SVM required for the applications
* **Output:** Saves an SVM in data/svm/svm_model.joblib

### `performance.py`

* **Paper Link:** Corresponds to **Section 5.1** and **Table 1**.
* **Purpose:** Model performance comparison
* **Output:** Outputs metrics of 5 different models and generates Results/model_comparison_results.json

### `topic_cluster.py`

* **Paper Link:** Corresponds to **Section 5.2** and **Table 2**.
* **Action:** Runs BERTopic to extract topic centroids. Applies LEACE (Linear Estimation and Concept Erasure) to project these centroids onto the null space of the partisan concept.
* **Output:** Generates the K-Means (k=20) clustering comparison before and after erasure, demonstrating the semantic repair of the "Democratic Party" split and the dissolution of the "Republican Composite".

### `subreddit_plot.py`

* **Paper Link:** Corresponds to **Section 5.3** and **Figure 1**.
* **Action:** Calculates the "Subreddit Centroids", applies LEACE to isolate the 1D partisan effect vector, and plots it against the percentage of texts classified as Republican by our SVM.
* **Output:** Generates `Results/subreddit_bias_plot.pdf`.

### `subreddit_cluster.py`

* **Paper Link:** Corresponds to **Section 5.4** and **Table 3**.
* **Action:** Clusters the subreddit centroids (k=6) before and after applying LEACE.
* **Output:** Generates the data for Table 3, revealing the "Political Discussion Merge" and the "Entity Focus Trap" (e.g., `r/democrats` migrating to the right-wing cluster due to shared entity focus).

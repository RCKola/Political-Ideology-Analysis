# PartisanNet: Untangling Political Ideology Representations in Online Communities

**Authors:** Thibault Cangemi & Zhiang Chen  
**Institution:** ETH Zürich  
**Course:** Supervised Research (Law, Economics, and Data Science)

## Overview

This repository serves as the official replication package for our paper, *"PartisanNet: Untangling Political Ideology Representations in Online Communities."* It contains all necessary code, data subsets, and instructions to reproduce the contrastive fine-tuning, LEACE concept erasure, and clustering results discussed in the text.

## 1. Repository Structure

```text
partisan_net_replication/
├── data/
│   ├── raw/                 # Scripts/instructions for full Kaggle datasets
│   └── processed/           # Contains the 50k comment subset used for experiments
├── models/                  # Pre-trained PartisanNet weights and svm_model.joblib
├── notebooks/               # Jupyter notebooks for reproducing paper figures/tables
├── src/                     # Source code for model training and utilities
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

* **Experiment Subset (Ready to Run):** The exact subset of 50,000 comments across 30 subreddits used for our primary experiments (Sections 5.2, 5.3, and 5.4) is provided in `data/processed/experiment_subset.csv`. **You can run the replication notebooks immediately using this file.**
* **Full Datasets (Optional):** The full "Public Opinion on Republicans (Rep-Ops)" and "Public Opinion on Democrats (Dem-Ops)" datasets can be downloaded from Kaggle (Asaniczka, 2024a; Asaniczka, 2024b).

## 4. Replication Guide

To reproduce the findings, tables, and figures from the paper, run the Jupyter Notebooks in the `notebooks/` directory in the following order:

### `01_train_partisannet.ipynb`

* **Purpose:** Demonstrates the contrastive fine-tuning process of the `all-MiniLM-L6-v2` backbone.
* **Note:** Full training takes approximately 50 epochs on an RTX 4060ti. For convenience, we have provided the pre-trained weights in the `/models/` directory so you can skip training and proceed directly to the analysis.

### `02_topic_modeling_and_erasure.ipynb`

* **Paper Link:** Corresponds to **Section 5.2** and **Table 2**.
* **Action:** Runs BERTopic to extract 81 topic centroids. Applies LEACE (Linear Estimation and Concept Erasure) to project these centroids onto the null space of the partisan concept.
* **Output:** Generates the K-Means (k=20) clustering comparison before and after erasure, demonstrating the semantic repair of the "Democratic Party" split and the dissolution of the "Republican Composite".

### `03_subreddit_alignment.ipynb`

* **Paper Link:** Corresponds to **Section 5.3** and **Figure 1**.
* **Action:** Calculates the "Subreddit Centroids", applies LEACE to isolate the 1D partisan effect vector, and plots it against the percentage of texts classified as Republican by our SVM.
* **Output:** Generates `results/figure_1_subreddit_alignment.png`.

### `04_community_realignment.ipynb`

* **Paper Link:** Corresponds to **Section 5.4** and **Table 3**.
* **Action:** Clusters the subreddit centroids (k=6) before and after applying LEACE.
* **Output:** Generates the data for Table 3, revealing the "Political Discussion Merge" and the "Entity Focus Trap" (e.g., `r/democrats` migrating to the right-wing cluster due to shared entity focus).

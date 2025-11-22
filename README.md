# Detecting Partisan Language in Reddit Communities

## Motivation: 
This project will address two primary research questions:

### Partisan Language Detection (RQ1)
Can we detect partisan language in Reddit based on posts and comments? The goal is to train a model to classify reddit comments or posts into left-leaning, neutral or right-leaning speech.

### Latent Space Analysis (RQ2)
Can the embeddings from the model identified in RQ1 be used to construct a latent embedding space? The goal of this space is twofold: (a) to successfully enable clustering of subreddits in a primarily unsupervised fashion, and (b) to use this space to analyze political lean of seemingly neutral communities by measuring their position relative to the established partisan poles.

## Background and Related Work: 
### RiCo (Reddit Ideological Communities)
The paper focused on classifying the ideological orientation of news articles using data from subreddits. By collecting and labeling news articles posted in “flagship” partisan subreddits, such as r/liberal and r/conservative, they found that a SVM model outperformed various Transformer models in accuracy.

### MBIB (Media Bias Identification Benchmark)
The paper introduces a unifying benchmark that groups different types of media bias, e.g. gender, political, etc., under a common framework to test detection techniques. They evaluated MBIB using various Transformer models, and they found that there is no single model which outperforms all the others for all biases simultaneously.

### S-BERT (Sentence-BERT)
This paper introduces S-BERT, a modification of BERT that uses siamese and triplet network structures to derive semantically meaningful embeddings. While BERT struggles for tasks like semantic similarity search and clustering, S-BERT drastically speeds up such tasks, while maintaining similar accuracy to BERT. 

## Dataset: 
The labeled training corpus will be a composite of two high-quality, documented sources.

Corpus 1 (Primary): The "Liberals vs Conservatives on Reddit [13000 posts]" dataset from Kaggle. This dataset consists of posts from partisan subreddits labeled either by “liberal”/”left” and “conservative”/”right”. For each post we are also given the amount of upvotes a post has, which can be used to quantify how much traction a post has obtained.

Corpus 2 (Secondary): The “mbib-base” dataset on Hugging Face. This dataset is a collection of 22 datasets used in bias-related publications. The datasets were manually filtered and assigned to the nine pre-defined MBIB tasks (e.g. Hate Speech, gender bias). Finally, the datasets were binarized such that each entry is either “biased” or “unbiased”.

We will use corpus 1 for RQ1 to train our model. However, as this only contains binary labels, we will complement this with Corpus 2, filtered for political bias, to be able to detect unbiased comments. The difficulty will be in creating a balanced dataset. There are multiple approaches we will try, for example using Corpus 1 to label the biased text in Corpus 2 to either “left” or “right”. 

## Methods:
After extracting and processing the relevant datasets, we will fine tune an S-BERT model to output embeddings for sentences / phrases, which is then fed into different MLP heads for a range of tasks including partisan language detection (RQ1). We claim that a specific loss formulation for this task leads to a unified political ideology latent representation that can be used downstream for community clustering (RQ2). This formulation could be inspired by recent advances in representation learning.

For RQ1 the model predictions will be evaluated on a hold out dataset on classification metrics such as accuracy, precision etc. While for RQ2 we will cluster the embeddings with HDBScan based on posts / comments from reddit communities. The latent space will then be analyzed on clustering metrics such as adjusted rand index, normalized mutual information, average silhouette width, graph connectivity etc. 

We will use a range of language modelling and traditional baselines for RQ1, namely a subset of the following methods: NaiveBayes, SVM, LGBM, DistilBERT, RoBERTa, GPT2, ULMfit, LongFormer, ConvBert. For RQ2 we will compare our embeddings against Word2Vec, TF-IDF and embeddings output from transformer models (see above). We will also visualize our results with dimensionality reduction methods like UMAP, t-SNE, PCA etc. and sample phrases or word clouds.

## Tables/Figures:
- Results tables from RQ1: classification metrics
- Results tables from RQ2: cluster metrics
- UMAP of clustered embeddings
- Word cloud from embeddings

## Timeline:
We will follow the timeline given in the course syllabus.


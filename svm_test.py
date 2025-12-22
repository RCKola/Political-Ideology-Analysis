from partisannet.data.datamodule import get_dataloaders
from sentence_transformers import SentenceTransformer
from partisannet.models.get_embeddings import generate_embeddings
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import torch
import numpy as np

def balance_tensors(embeddings, labels):
    """
    Subsamples embeddings and labels to have the same number of samples per class.
    Returns balanced tensors.
    """
    # 1. Convert to numpy for easier indexing
    labels_np = labels.cpu().numpy()
    unique_classes, counts = np.unique(labels_np, return_counts=True)
    
    # 2. Find the size of the smallest class
    min_count = counts.min()
    print(f"Balancing to {min_count} samples per class...")
    
    selected_indices = []
    
    # 3. Sample indices for each class
    for cls in unique_classes:
        # Find where the label equals this class
        indices = np.where(labels_np == cls)[0]
        
        # Randomly pick 'min_count' indices
        sampled = np.random.choice(indices, min_count, replace=False)
        selected_indices.extend(sampled)
    
    # 4. Shuffle the results so classes are mixed
    np.random.shuffle(selected_indices)
    
    # 5. Slice the original tensors
    balanced_embeddings = embeddings[selected_indices]
    balanced_labels = labels[selected_indices]
    
    return balanced_embeddings, balanced_labels

def svm_report(
        embeddings_train,
        labels_train,
        labels_test,
        embeddings_test,
        clf_tot = SVC(kernel='linear', random_state=42)
):
    clf_tot.fit(embeddings_train, labels_train)
    predictions_tot = clf_tot.predict(embeddings_test)

    print("Classification Report")
    print(classification_report(labels_test, predictions_tot))

    accuracy_tot = accuracy_score(labels_test, predictions_tot)
    print(f"Accuracy: {accuracy_tot:.4f}")

def  svm_test(
        embeddings,
        labels
):
    
    print("Whole Sample")
    embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    svm_report(
        embeddings_train,
        labels_train,
        labels_test,
        embeddings_test
    )
    print("Balanced Sample with ROS")
   

    ros = RandomOverSampler(random_state=42)
    embeddings_resampled, labels_resampled = ros.fit_resample(embeddings, labels)
    embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(embeddings_resampled, labels_resampled, test_size=0.2, random_state=42)

    svm_report(
        embeddings_train,
        labels_train,
        labels_test,
        embeddings_test
    )

    print("Balanced Sample with Undersampling")

    balanced_embeddings, balanced_labels = balance_tensors(embeddings, labels)
    embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(balanced_embeddings, balanced_labels, test_size=0.2, random_state=42)

    svm_report(
        embeddings_train,
        labels_train,
        labels_test,
        embeddings_test
    )


if __name__ == "__main__":
    #show_topics()
    #cache_path = "./cached_processed_dataset"


    dataloaders, topic_model = get_dataloaders("LibCon", batch_size=32, split=False, num_topics=None, cluster_in_k=40, renew_cache=False)
    
    
    
    embeddings, partisan_labels, old_topics = generate_embeddings(dataloaders['train'], path = "data/fine_tuned_sbert")

    svm_test(
        embeddings,
        partisan_labels
    )
    print("SVM Test Completed.")
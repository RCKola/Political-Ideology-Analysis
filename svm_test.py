from partisannet.data.datamodule import get_dataloaders
from partisannet.models.get_embeddings import generate_embeddings
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import joblib




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

def  svm_train(
        embeddings,
        labels
):
    
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(embeddings, labels)
    joblib.dump(clf, "data/svm/svm_model.joblib")
   


if __name__ == "__main__":
   
    train = True

    if train:
        dataloaders = get_dataloaders("DemRep", batch_size=32, split=False, renew_cache=False)
        embeddings, partisan_labels, _ = generate_embeddings(dataloaders['train'], path = "data/centerloss_sbert_full")
        svm_train(embeddings, partisan_labels)



    dataloaders = get_dataloaders("testdata",batch_size=32, split=False, renew_cache=False)
    
    

    embeddings, partisan_labels, _ = generate_embeddings(dataloaders['train'], path = "data/centerloss_sbert_full")
    clf_tot = joblib.load("data/svm/svm_model.joblib")
    predictions_tot = clf_tot.predict(embeddings)

    print("Classification Report")
    print(classification_report(partisan_labels, predictions_tot))

    accuracy_tot = accuracy_score(partisan_labels, predictions_tot)
    print(f"Accuracy: {accuracy_tot:.4f}")
    
    print("SVM Test Completed.")
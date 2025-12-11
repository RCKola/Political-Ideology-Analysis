import os
import torch
from concept_erasure import LeaceEraser
from partisannet.data.datamodule import get_dataloaders
from partisannet.models.classifier import SBERTClassifier
from partisannet.data.datamodule import include_topics, load_datasets
from partisannet.modelling import PartisanNetModel 
from partisannet.models.get_embeddings import generate_embeddings
from datasets import load_from_disk



if __name__ == "__main__":
    #show_topics()
    cache_path = "./cached_processed_dataset"
    embeddings_cache_path = "./cached_embeddings.pt"

    # 2. Check if cache exists
    if os.path.exists(cache_path) and os.path.exists(embeddings_cache_path):
        print("Loading dataset and embeddings from disk...")
        dataset = load_from_disk(cache_path)
        final_embeddings = torch.load(embeddings_cache_path)
    else:
        print("Cache not found. Processing data...")
        # --- Your Expensive Logic Here ---
        dataloaders = get_dataloaders("LibCon", batch_size=32, split=False)
        
        # Generate Embeddings
        final_embeddings = generate_embeddings(dataloaders['train'])
        
        # Process Dataset (Map/Filter)
        dataset = load_datasets("LibCon")
        dataset, topic_model = include_topics(dataset, remove_stopwords=True)
        
        # --- Save to Disk for Next Time ---
        print("Saving dataset and embeddings to disk...")
        dataset.save_to_disk(cache_path)
        torch.save(final_embeddings, embeddings_cache_path)


    print(f"Extracted embeddings shape: {final_embeddings.shape}")
  
    y = (torch.tensor(dataset['topic']) == -1).long()
    
    eraser = LeaceEraser.fit(final_embeddings, y)
    erased_embeddings = eraser(final_embeddings)
    print(f"Erased embeddings shape: {erased_embeddings.shape}")
    
"""     print(dataset.column_names)
    print("Topic Info:", topic_model.get_topic_info())
    print("Topic labels:", topic_model.generate_topic_labels(nr_words=1))
    print("Sample topics from dataset:")
    for i in range(5):
        print(f"Doc: {dataset[i]['text'][:50]}... Topic: {dataset[i]['topic']}, Prob: {dataset[i]['topic_prob']}") """
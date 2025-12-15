import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # For a nice progress bar



def generate_embeddings(dataloader, model = SentenceTransformer('all-MiniLM-L6-v2')):
    all_embeddings = []
    all_labels = []
    all_topics = []
    # Switch model to eval mode (good practice, though SBERT handles this largely internally)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Processing on: {device}")
    
    # Wrap loader in tqdm for a progress bar
    for batch in tqdm(dataloader, desc="Encoding sentences"):
        
        # NOTE: Inspect your 'batch' variable here! 
        # If your loader returns a dict, access the text key: sentences = batch['text']
        # If your loader returns a tuple (text, label), access text: sentences = batch[0]
        # In this simple example, the batch is just a list of strings.
        sentences = batch["text"]
        labels = batch.get("label", None)  # Optional, if labels are provided
        topic = batch.get("topic", None)  # Optional, if topics are provided
        with torch.no_grad():
            # convert_to_tensor=True returns a PyTorch tensor instead of a numpy array
            batch_embeddings = model.encode(
                sentences, 
                convert_to_tensor=True, 
                show_progress_bar=False
            )
            
            # Move to CPU to avoid filling up GPU memory if the dataset is huge
            all_embeddings.append(batch_embeddings.cpu())
            if labels is not None:
                all_labels.extend(labels)
            if topic is not None:
                all_topics.extend(topic)
    # Concatenate all batches into one large tensor (N_samples x Embedding_Dimension)
    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_labels = torch.tensor(all_labels) if all_labels else None
    final_topics = torch.tensor(all_topics) if all_topics else None
    return final_embeddings, final_labels, final_topics

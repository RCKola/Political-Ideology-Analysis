import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # For a nice progress bar



def generate_embeddings(dataloader, path = 'all-MiniLM-L6-v2'):
    all_embeddings = []
    all_labels = []
    all_subreddits = []
    # Switch model to eval mode (good practice, though SBERT handles this largely internally)
    model = SentenceTransformer(path)
    
    
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
        subreddits = batch.get("subreddit", None)  # Optional, if subreddit info is provided
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
            if subreddits is not None:
                all_subreddits.extend(subreddits)
            
    # Concatenate all batches into one large tensor (N_samples x Embedding_Dimension)
    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_labels = torch.tensor(all_labels) if all_labels else None
    final_subreddits = all_subreddits if all_subreddits else None
    
    return final_embeddings, final_labels, final_subreddits

def get_finetuned_embeddings(dataset, model_path="data/fine_tuned_sbert"):
    """
    Loads your custom model and extracts embeddings from the last SBERT layer.
    """
    print(f"Loading Fine-Tuned model from: {model_path}")
    model = SentenceTransformer(model_path)
    
    # Switch to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Extract text and labels
    texts = dataset['text'] # Assuming dataset is a Hugging Face dataset
    labels = dataset['label']
    topics = dataset['topics']
    
    print("Encoding embeddings...")
    # encode() automatically handles batching and tokenization
    # convert_to_tensor=True gives you a PyTorch tensor on the correct device
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    return embeddings, torch.tensor(labels), torch.tensor(topics) if topics is not None else None
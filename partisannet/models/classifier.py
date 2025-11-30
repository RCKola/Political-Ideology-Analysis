import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class SBERTClassifier(nn.Module):
    def __init__(self, model_name='all-MiniLM-L6-v2', num_classes=2):
        super(SBERTClassifier, self).__init__()
        self.sbert = SentenceTransformer(model_name)
        self.embed_dim = self.sbert.get_sentence_embedding_dimension()
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        print("SBERT modules:", (module for module in self.sbert.named_children()))

    def forward(self, sentences):
        # 1. Tokenize the sentences (This creates tensors on the CPU)
        data_dict = self.sbert.tokenize(sentences)
        
        # 2. FIX: Move every tensor in the dictionary to the same device as the model (GPU)
        device = next(self.parameters()).device # specific trick to find where the model is
        
        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device)
        
        # 3. Now feed the GPU tensors into the GPU model
        output_dict = self.sbert(data_dict)
        
        # The rest of your code remains the same...
        embeddings = output_dict['sentence_embedding']
        logits = self.classifier(embeddings)
        return logits
    
    @torch.inference_mode()
    def predict(self, sentences):
        logits = self.forward(sentences)
        probabilities = F.softmax(dim=1)(logits)
        return probabilities


if __name__ == "__main__":
    model = SBERTClassifier()
    print(model)
    
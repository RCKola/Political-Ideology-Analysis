import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class SBERTClassifier(nn.Module):
    def __init__(self, model_name='all-MiniLM-L6-v2', num_classes=2, freeze_backbone=False, dropout_prob=0.3):
        super(SBERTClassifier, self).__init__()
        self.sbert = SentenceTransformer(model_name)
        if freeze_backbone:
                    for param in self.sbert.parameters():
                        param.requires_grad = False
                # --------------------

        self.embed_dim = self.sbert.get_sentence_embedding_dimension()
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        print("SBERT modules:", (module for module in self.sbert.named_children()))

    def forward(self, sentences, return_embeddings=False):
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


        if return_embeddings:
            return embeddings
        logits = self.classifier(embeddings)
        return logits, embeddings
    
    @torch.inference_mode()
    def predict(self, sentences):
        logits = self.forward(sentences)[0]
        probabilities = F.softmax(dim=1)(logits)
        return probabilities


if __name__ == "__main__":
    model = SBERTClassifier()
    print(model)
    
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
        data_dict = self.sbert.tokenize(sentences)
        output_dict = self.sbert(data_dict)
        # embeddings = F.normalize(embeddings, p=2, dim=1)
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
    
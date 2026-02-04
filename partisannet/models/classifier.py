import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models
from peft import LoraConfig, TaskType


class SBERTClassifier(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', num_classes=2, freeze_backbone=False, lora_r=64):
        super(SBERTClassifier, self).__init__()
        self.sbert = SentenceTransformer(model_name)
        
        # transformer = models.Transformer(model_name)
        # pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="max")
        # self.sbert = SentenceTransformer(modules=[transformer, pooling])

        if freeze_backbone:
            for param in self.sbert.parameters():
                param.requires_grad = False

        self.embed_dim = self.sbert.get_sentence_embedding_dimension()
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        print("SBERT modules:", [module for module in self.sbert.named_children()])
        if lora_r is not None and lora_r > 0:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                target_modules=["query", "value"],
                r=lora_r,
                lora_alpha=2*lora_r,
                lora_dropout=0.1
            )
            self.sbert.add_adapter(peft_config)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, sentences):
        data_dict = self.sbert.tokenize(sentences)
        device = self.device
        
        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device)
        
        output_dict = self.sbert(data_dict)
        embeddings = output_dict['sentence_embedding']

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
    
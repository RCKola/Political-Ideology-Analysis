import argparse

import torch
import pytorch_lightning as L
from partisannet.modelling import PartisanNetModel, save_model

from partisannet.data.datamodule import get_dataloaders
from partisannet.models.classifier import SBERTClassifier
from partisannet.utils.training import setup_logger, setup_callbacks

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def run_training(args):
    print(f"Training with {args.max_epochs} epochs...")

    log_dir = args.log_dir
    model_name = args.model_name
    model_dir = f"data/{model_name.replace('/', '_')}_finetuned"
    
    dataloaders, topic_model = get_dataloaders(
        "DemRep", 
        batch_size=args.batch_size, 
        split=True, 
        num_topics=None, 
        cluster_in_k=40, 
        renew_cache=True
    )
    sbert_model = SBERTClassifier(lora_r=args.lora_r)
    model = PartisanNetModel(sbert_model, lr=args.lr)
    logger = setup_logger(log_dir)
    logger.watch(model, log="all")

    L.seed_everything(42, workers=True)
    trainer = L.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=args.max_epochs, 
        logger=logger,
        # callbacks=setup_callbacks(log_dir),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    results = trainer.test(model, dataloaders['test'], ckpt_path="best")

    save_model(trainer, model, path=model_dir)
    return model_dir, results

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default="data/outputs", help='Directory to save logs and model checkpoints')
    parser.add_argument('--model_name', type=str, default="sentence-transformers/all-MiniLM-L6-v2", help='Pretrained SBERT model name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank for adapter layers')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for the optimizer')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    run_training(args)

if __name__ == "__main__":
    main()
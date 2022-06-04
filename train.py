from boto import config
from datasets import load_from_disk,DatasetDict
import datasets
from transformers import AutoTokenizer, TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification
from pathlib import Path
from sklearn.metrics import log_loss
import torch
import torch.nn.functional as F
import argparse
def score(preds):
    return {'log loss': log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}
def getTrainer(args,dataDict,tokenizer):
    train_args = TrainingArguments(
            'outputs',
            learning_rate=args.learning_rate,
            warmup_ratio=0.1, 
            lr_scheduler_type='cosine',
            fp16=False,
            evaluation_strategy='epoch',
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size*2,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            report_to='none'
        )
    
    model = AutoModelForSequenceClassification.from_pretrained(args.modelPath, num_labels=3)
    
    return Trainer(model, 
                   train_args, 
                   train_dataset=dataDict['train'],
                   eval_dataset=dataDict['test'], 
                   tokenizer=tokenizer,
                   compute_metrics=score)
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.modelPath)
    datapath = Path(args.data_path)
    
    train_data = load_from_disk(datapath/"train")
    val_data = load_from_disk(datapath/"val")

    dataDict = DatasetDict({
        "train":train_data,
        "test":val_data
    })

    trainer = getTrainer(args,dataDict,tokenizer)

    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(args.model_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str,default=".")
    parser.add_argument("--learning_rate",type=int,default=1e-5)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--weight_decay",type=float,default=0.01)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--modelPath",type=str,default="microsoft/deberta-v3-small")
    parser.add_argument("--model_dir",type=str,default="/home/public/yushilin/nlp/feedback")
    args = parser.parse_args()
    main(args)
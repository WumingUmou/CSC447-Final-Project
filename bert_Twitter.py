import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

RANDOM_SEED = 42
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score as auroc

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
import re
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import sys

print("IMPORT FINISHED")

infile = sys.argv[1]
outfile = sys.argv[2]
N_EPOCHS = int(sys.argv[3])
BATCH_SIZE = int(sys.argv[4])
max_token_len = int(sys.argv[5])



data = pd.read_csv(infile, index_col=0)

# Data Preprocessing

def clean_str(s):
    regs = [
        r'(\/\/www[^\s]+)',
        r'(pic.twitter.com\/[^\s]+)',
        r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)',
        r'(?i)\b((?:https?:\/\/[^\s]+))',
        r'https',
    ]
    
    cleaned_s=s.replace("RT", "")
    cleaned_s=cleaned_s.replace("&amp", "")
    cleaned_s=cleaned_s.replace("\\n", "")
    cleaned_s = " ".join([x[:-2] for x in cleaned_s.split("\\x")])
    cleaned_s=cleaned_s.lower()[1:]
    
    for reg in regs:
        cleaned_s = re.sub(reg, " ", cleaned_s)
    return cleaned_s

data["retweet"] = data["full_text"].str.slice(2,4)=="RT"
data["cleaned_text"]=data["full_text"].apply(clean_str)

data.fillna(" ", inplace=True)
data["party_R"]=data["party_id"].apply(lambda x: 0 if x=="D" else 1)
data["party_D"]=data["party_id"].apply(lambda x: 0 if x=="R" else 1)
train_df = data.sample(frac=0.8)
val_df = data.drop(train_df.index)

predict_df = pd.read_csv("test.csv")
predict_df = predict_df.fillna("")
predict_df["retweet"] = predict_df["full_text"].str.slice(2,4)=="RT"
predict_df["cleaned_text"]=predict_df["full_text"].apply(clean_str)

print("DATA PREPROCESSED")

# Network Initialize

BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

## Create Dataset for BERT model

class TwitterDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: BertTokenizer,
                 max_token_len: int = max_token_len
                ):
        
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row.cleaned_text
        label1, label2 = 0, 0
        if "party_id" in data_row:
            label1 = data_row.party_R
            label2 = data_row.party_D
        encoding = self.tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=self.max_token_len,
                        return_token_type_ids=False,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                    )
        

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            
            labels=torch.FloatTensor(np.array([label1, label2]))
        )

train_dataset = TwitterDataset(data, tokenizer, max_token_len)

## Load pretrained weights

bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)

## Create DataModule (Combination of DataLoaders)

class TwitterDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, predict_df, tokenizer, batch_size=BATCH_SIZE, max_token_len=max_token_len):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.predict_df = predict_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = TwitterDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )
        self.val_dataset = TwitterDataset(
            self.val_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )



data_module = TwitterDataModule(
    train_df,
    val_df,
    predict_df,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=max_token_len
)

# Define Training/Testing/Validation steps

class TwitterTagger(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def predict_dataloader(self):
        return DataLoader(
            TwitterDataset(
                predict_df,
                tokenizer,
                max_token_len
            ),
            batch_size = BATCH_SIZE
        )
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=BATCH_SIZE)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=BATCH_SIZE)
        return loss

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        _, preds = self(input_ids, attention_mask,)
        return preds.cpu().detach().numpy()

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        
        for i, name in enumerate(["party_R", "party_D"]):
            class_roc_auc = auroc(labels[:, i].numpy(), (predictions[:, i].numpy()>0.5).astype(int))
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

# Calculate other numbers

steps_per_epoch=len(train_df) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS

warmup_steps = total_training_steps // 5
warmup_steps, total_training_steps

## Create model

model = TwitterTagger(
  n_classes=2,
  n_warmup_steps=warmup_steps,
  n_training_steps=total_training_steps
)

## Define callbacks

checkpoint_callback = ModelCheckpoint(
  dirpath="checkpoints",
  filename="best-checkpoint",
  save_top_k=2,
  verbose=True,
  monitor="val_loss",
  mode="min"
)
early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min")

logger = TensorBoardLogger("lightning_logs", name="Twitter")

# Initialize trainer

trainer = pl.Trainer(
  logger=logger,
  checkpoint_callback=[checkpoint_callback,early_stopping_callback],
  max_epochs=N_EPOCHS,
  gpus=1,
  progress_bar_refresh_rate=30,
)

# Model Fitting

trainer.fit(model, data_module)

# Model Prediction

trained_model = TwitterTagger.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, n_classes=2)

trained_model.eval()
trained_model.freeze()

result = trainer.predict()
result = np.concatenate(result, axis=0)

result = ["R" if x else "D" for x in result[:,0]>result[:,1]]

pd.DataFrame({"party":result}).reset_index().rename(columns={"index":"id"}).to_csv(outfile, index=False)

print("Best model saved at", trainer.checkpoint_callback.best_model_path)
print("Prediction result saved at", outfile)
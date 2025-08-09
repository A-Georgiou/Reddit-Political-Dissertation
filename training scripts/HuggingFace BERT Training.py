# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:50:09 2021

@author: Andre
"""

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import Trainer, TrainingArguments

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.ERROR)

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

class SentimentClassifier(nn.Module):
      def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.pre_classifier = nn.Linear(768, 768)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
      def forward(self, input_ids, attention_mask):
        pool_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = pool_output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.drop(pooler)
        output = self.out(pooler)
        return output
    
class GPReviewDataset(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
    
  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      return_token_type_ids=False,
      max_length = 256,
      padding = 'max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

class ModelHandler:
	def eval_model(model, data_loader, loss_fn, device, n_examples):
	  model = model.eval()
	  losses = []
	  correct_predictions = 0
	  with torch.no_grad():
	    for d in data_loader:
	        input_ids = d["input_ids"].to(device)
	        attention_mask = d["attention_mask"].to(device)
	        targets = d["targets"].to(device)
	        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
	        _, preds = torch.max(outputs, dim=1)
	        loss = loss_fn(outputs, targets)
	        correct_predictions += torch.sum(preds == targets)
	        losses.append(loss.item())
	  return correct_predictions.double() / n_examples, np.mean(losses)
    
	def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
	    model = model.train()
	    losses = []
	    correct_predictions = 0
	    for i, d in tqdm(enumerate(data_loader)):
	        optimizer.zero_grad()
	        input_ids = d["input_ids"].to(device)
	        attention_mask = d["attention_mask"].to(device)
	        targets = d["targets"].to(device)
	        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
	        loss = loss_fn(outputs, targets)
	        
	        _, preds = torch.max(outputs, dim=1)
	        
	        if i%5000==0:
	            print(f'Epoch: {epoch}, Loss: {loss.item()}')
	            
	        correct_predictions += torch.sum(preds == targets)
	        
	        losses.append(loss.item())
	        loss.backward()
	        optimizer.step()
	    return correct_predictions.double() / n_examples, np.mean(losses)
  

def create_data_loader(df, tokenizer, max_len, batch_size):
        ds = GPReviewDataset(reviews=df.comment.to_numpy(), targets=df.sentiment.to_numpy(),tokenizer=tokenizer,max_len=max_len)        
        return DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=True)

if __name__== '__main__':
    RANDOM_SEED = 42
    MAX_LEN = 256
    EPOCHS = 20
    
    batch_sizes = [12]
    learning_rates = [3e-5]

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    df = pd.read_csv('datasets/combined_output.csv')
    
    modelHandler = ModelHandler()

    class_names = ['far-left', 'centre-left', 'centre-right', 'far-right']
    
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, truncation=True, do_lower_case=True)
    best_accuracy = 0
    
    for batch_s in batch_sizes:
        print("NEW BATCH SIZE: ", batch_s)
        for learn_r in learning_rates:
            print("NEW LEARNING RATE: ", learn_r)
            df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
            df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
            
            
            train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, batch_s)
            val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, batch_s)
            test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, batch_s)
            
            data = next(iter(train_data_loader))
            print(data.keys())
            
            print(data['input_ids'].shape)
            print(data['attention_mask'].shape)
            print(data['targets'].shape)
            
            bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
            
            #albert = bert_model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
            #print(albert['last_hidden_state'].shape)
            
            model = SentimentClassifier(len(class_names))
            model = model.to(device)
            
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            
            print(input_ids.shape) # batch size x seq length
            print(attention_mask.shape) # batch size x seq length
            
            optimizer = AdamW(model.parameters(), lr=learn_r)
            total_steps = len(train_data_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
            loss_fn = nn.CrossEntropyLoss().to(device)
        
            history = defaultdict(list)
            

            for epoch in range(EPOCHS):
              print(f'Epoch {epoch + 1}/{EPOCHS}')
              print('-' * 10)
              train_acc, train_loss = modelHandler.train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
              print(f'Train loss {train_loss} accuracy {train_acc}')
              val_acc, val_loss = modelHandler.eval_model(model, val_data_loader, loss_fn, device, len(df_val))
              print(f'Val   loss {val_loss} accuracy {val_acc}')
              print()
              history['train_acc'].append(train_acc)
              history['train_loss'].append(train_loss)
              history['val_acc'].append(val_acc)
              history['val_loss'].append(val_loss)
              if val_acc > best_accuracy:
                  torch.save(model.state_dict(), 'best_model_multi_class.bin')
                  tokenizer.save_vocabulary('distilbert_vocab_multi.bin')
                  best_accuracy = val_acc
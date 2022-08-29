from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import pandas as pd

class BERTDataset:
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        text = str(self.data[item]['text'])
        label = self.data[item]['label']
        
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, pad_to_max_length=True)
        
        input_ids = encoding['input_ids']
        token_type_ids = encoding['token_type_ids']
        attention_mask = encoding['attention_mask']
        
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'label': label}
    
    def collate_fn(self, batch):
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        token_type_ids = torch.stack([torch.tensor(item['token_type_ids']) for item in batch])
        attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
        label = torch.stack([torch.tensor([item['label']]) for item in batch])
        
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'label': label}
    
    def get_labels(self):
        return list(set([item['label'] for item in self.data]))
    
    def get_label_map(self):
        return {label: i for i, label in enumerate(self.get_labels())}
    
    def get_label_from_map(self, label_map):
        return {i: label for label, i in label_map.items()}



class ToxicModel(nn.module):
    """A simple bert model for training a 2 class  classification"""


    def __init__(self, bert_model, num_labels):
        super(ToxicModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_model, num_labels=num_labels)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert(input_ids, token_type_ids, attention_mask)
    
    def predict(self, input_ids, token_type_ids, attention_mask):
        return self.forward(input_ids, token_type_ids, attention_mask)
    
    def predict_batch(self, input_ids, token_type_ids, attention_mask):
        return self.forward(input_ids, token_type_ids, attention_mask)
    
    def predict_proba(self, input_ids, token_type_ids, attention_mask):
        return self.forward(input_ids, token_type_ids, attention_mask)
    
    def predict_proba_batch(self, input_ids, token_type_ids, attention_mask):
        return self.forward(input_ids, token_type_ids, attention_mask)
    
    def predict_proba_batch_with_labels(self, input_ids, token_type_ids, attention_mask):
        return self.forward(input_ids, token_type_ids, attention_mask)
    
    def predict_proba_with_labels(self, input_ids, token_type_ids, attention_mask):
        return self.forward(input_ids, token_type_ids, attention_mask)
    
    def predict_proba_with_labels_batch(self, input_ids, token_type_ids, attention_mask):
        return self.forward(input_ids, token_type_ids, attention_mask)
    
    def predict_proba_batch_with_labels(self, input_ids, token_type_ids, attention_mask):
        return self.forward(input_ids, token_type_ids, attention_mask)   


    def train(model, train_data, valid_data, epochs, batch_size, lr, device):   
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn)
        valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, collate_fn=valid_data.collate_fn)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            for batch in train_dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, token_type_ids, attention_mask, label = batch
                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = loss_fn(outputs, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                valid_acc = 0
                for batch in valid_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, token_type_ids, attention_mask, label = batch
                    outputs = model(input_ids, token_type_ids, attention_mask)
                    loss = loss_fn(outputs, label)
                    valid_loss += loss.item()
                    valid_acc += (outputs.max(1)[1] == label).sum().item()
                print(f'Epoch: {epoch+1:02} | Train Loss: {loss.item():.3f} | Train Acc: {(outputs.max(1)[1] == label).sum().item()/len(label):.3f} | Val. Loss: {valid_loss/len(valid_data):.3f} | Val. Acc: {valid_acc/len(valid_data):.3f}')    
        return model  

    def evaluate(model, data, batch_size, device):
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn)
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_acc = 0
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, token_type_ids, attention_mask, label = batch
                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = loss_fn(outputs, label)
                valid_loss += loss.item()
                valid_acc += (outputs.max(1)[1] == label).sum().item()
            return valid_loss/len(data), valid_acc/len(data)

    def evaluate_batch(model, data, batch_size, device):
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn)
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_acc = 0
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, token_type_ids, attention_mask, label = batch
                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = loss_fn(outputs, label)
                valid_loss += loss.item()
                valid_acc += (outputs.max(1)[1] == label).sum().item()
            return valid_loss/len(data), valid_acc/len(data)

    def evaluate_batch_with_labels(model, data, batch_size, device):
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn)
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_acc = 0
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, token_type_ids, attention_mask, label = batch
                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = loss_fn(outputs, label)
                valid_loss += loss.item()
                valid_acc += (outputs.max(1)[1] == label).sum().item()
            return valid_loss/len(data), valid_acc/len(data)


    def evaluate_with_labels(model, data, batch_size, device):
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn)
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_acc = 0
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, token_type_ids, attention_mask, label = batch
                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = loss_fn(outputs, label)
                valid_loss += loss.item()
                valid_acc += (outputs.max(1)[1] == label).sum().item()
            return valid_loss/len(data), valid_acc/len(data)



    










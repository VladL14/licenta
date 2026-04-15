import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from dataset import DatasetML

MAX_LEN = 64
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.30
tokenizer = DistilBertTokenizer.from_pretrained('backend+ml/model_output')
model = DistilBertForSequenceClassification.from_pretrained('backend+ml/model_output', num_labels=5)
model.to(DEVICE)
model.eval()

test_dataset = DatasetML("backend+ml/data/test.tsv", tokenizer, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
all_preds = []
all_targets = []
print("Incepere evaluare...")
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['ids'].to(DEVICE)
        attention_mask = batch['mask'].to(DEVICE)
        targets = batch['targets'].numpy()
        
        outputs = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(outputs.logits).cpu().numpy()
        predictions = (probabilities >= THRESHOLD).astype(int)
        
        all_preds.extend(predictions)
        all_targets.extend(targets)

target_names = ['Happy', 'Sad', 'Angry', 'Surprised', 'Anxious']

print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))
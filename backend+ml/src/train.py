import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from dataset import DatasetML

MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_dataset = DatasetML("backend+ml/data/train.tsv", tokenizer, max_len=MAX_LEN)
val_dataset = DatasetML("backend+ml/data/dev.tsv", tokenizer, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
model.to(DEVICE)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for i, batch in enumerate(train_loader):
        input_ids = batch['ids'].to(DEVICE)
        attention_mask = batch['mask'].to(DEVICE)
        targets = batch['targets'].to(DEVICE).float()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs.logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['ids'].to(DEVICE)
            attention_mask = batch['mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE).float()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs.logits, targets)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
output_dir = './model_output'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Training complete.")

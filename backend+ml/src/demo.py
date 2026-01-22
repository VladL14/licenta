from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DistilBertForSequenceClassification.from_pretrained('./model_output')
tokenizer = DistilBertTokenizer.from_pretrained('./model_output')

model.to(DEVICE)
model.eval()

while True:
    text = input("Enter text (or 'exit' to quit): ")
    if text == 'exit':
        break
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoding['ids'].to(DEVICE)
    attention_mask = encoding['mask'].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(outputs.logits)
        emotions = ["happy", "sad", "angry", "surprised", "anxious"]
        for index in range(len(emotions)):
            if probabilities[0][index] >= 0.5:
                print(f"{emotions[index]}: {probabilities[0][index]:.4f}")

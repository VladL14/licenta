from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import re
import numpy as np

THRESHOLD = 0.30
DELTA = 0.05
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DistilBertForSequenceClassification.from_pretrained('backend+ml/model_output', num_labels=5)
tokenizer = DistilBertTokenizer.from_pretrained('backend+ml/model_output')

model.to(DEVICE)
model.eval()

def invalid_input(text):
    text = text.strip()
    if len(text) < 3:
        return True
    if re.fullmatch(r'[\W_]+', text):
        return True
    if re.fullmatch(r'(.)\1+', text):
        return True
    return False

def decide_emotion(probabilities):
    probs = probabilities.cpu().numpy()[0]
    sorted_indices = np.argsort(probs)[::-1]
    p1 = probs[sorted_indices[0]]
    p2 = probs[sorted_indices[1]]
    emotions = ["happy", "sad", "angry", "surprised", "anxious"]
    top_emotion = emotions[sorted_indices[0]]
    if p1 < THRESHOLD:
        if abs(p1 - p2) < DELTA:
            return "confused"
        return "neutral"
    if abs(p1 - p2) < DELTA:
        return "confused"
    return top_emotion

while True:
    text = input("Enter text (or 'exit' to quit): ")
    if text == 'exit':
        break
    if invalid_input(text):
        print("Final emotion: neutral")
        continue
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(outputs.logits)
        emotions = ["happy", "sad", "angry", "surprised", "anxious"]
        for index in range(len(emotions)):
            print(f"{emotions[index]}: {probabilities[0][index].item() * 100:.2f}%")
        final_emotion = decide_emotion(probabilities)
        print(f"Final emotion: {final_emotion}")
            

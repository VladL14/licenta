from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import re
import numpy as np
import os
from dotenv import load_dotenv
import replicate

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(CURRENT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    print("nu a fost gasit api token in fisierul env")
else:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

class Message(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
    text: str
    image_code: str
    history: List[Message] = []
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

THRESHOLD = 0.30
DELTA = 0.05
model = DistilBertForSequenceClassification.from_pretrained('./model_output', num_labels=5)
tokenizer = DistilBertTokenizer.from_pretrained('./model_output')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    # if p1 < THRESHOLD:
    #     if abs(p1 - p2) < DELTA:
    #         return "confused"
    #     return "neutral"
    # if abs(p1 - p2) < DELTA:
    #     return "confused"
    if abs(p1-p2) < DELTA:
        if p1 < THRESHOLD:
            return "neutral"
        return "confused"
    if p1 < THRESHOLD:
        return "neutral"
    return top_emotion

@app.post("/chat")
def analyze_text(request: ChatInput):
    text = request.text
    image_code = request.image_code
    history = request.history
    if invalid_input(text):
        return {"text": text, "dominant_emotion": "neutral", "emotions_array": {}, "generated_image": image_code, "bot_response": "Please provide a valid input text."}
    encoding = tokenizer(
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
        final_emotion = decide_emotion(probabilities)
        emotions = ["happy", "sad", "angry", "surprised", "anxious"]
        emotions_array = {}
        for index in range(len(emotions)):
            emotions_array[emotions[index]] = round(probabilities[0][index].item() * 100, 2)
    bot_response = ""
    try:
        formatted_history = ""
        for message in history:
            if message.role == "user":
                formatted_history += f"User: {message.content}\n"
            else:
                formatted_history += f"Bot: {message.content}\n"

        current_prompt = f"{formatted_history}User: {text}\nBot:"

        system_prompt = (
                "You are an empathetic, human-like digital avatar engaged in a continuous conversation with a user. "
                f"Based on the user's message, you should feel and act: {final_emotion.upper()}. "
                "Read the conversation history to understand the context."
                "Respond directly and naturally in 1 or 2 sentances. "
                "CRITICAL: Always end your response with an open-ended question to keep the user engaged and prompt them to share text from which emotions can be detected."
            )
        llama_output = replicate.run("meta/meta-llama-3-8b-instruct",
                input={
                    "system_prompt": system_prompt,
                    "prompt": current_prompt,
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            )
        bot_response = "".join(llama_output)
    except Exception as e:
        print(f"Error generating bot response: {e}")
        bot_response = "Sorry, I can't respond right now."
    generated_image = None
    prompts = {
            "happy": (
                "Edit only the facial expression: the person is genuinely happy and joyful. "
                "Show a natural, warm smile with slightly raised cheeks, soft crow's feet wrinkles around the eyes, "
                "and eyes that are slightly squinted from smiling. "
                "Do not change anything else: background, lighting, clothing, hair, skin tone, face shape, camera angle."
            ),
            "sad": (
                "Edit only the facial expression: the person looks deeply sad and melancholic. "
                "Inner corners of the eyebrows raised and pulled together, upper eyelids drooping, "
                "mouth corners pulled down, lower lip slightly pushed up. Glassy, downcast eyes. "
                "Do not change anything else: background, lighting, clothing, hair, skin tone, face shape, camera angle."
            ),
            "angry": (
                "Edit only the facial expression: the person looks intensely angry and furious. "
                "Eyebrows sharply pulled down and together forming deep frown lines, upper eyelids tense and lowered, "
                "nostrils slightly flared, jaw clenched, lips pressed firmly together. "
                "Do not change anything else: background, lighting, clothing, hair, skin tone, face shape, camera angle."
            ),
            "surprised": (
                "Edit only the facial expression: the person looks completely shocked and surprised. "
                "Eyebrows raised as high as possible, forehead wrinkled, eyes wide open with visible white above the iris, "
                "jaw dropped with mouth open in a wide O shape. "
                "Do not change anything else: background, lighting, clothing, hair, skin tone, face shape, camera angle."
            ),
            "anxious": (
                "Edit only the facial expression: the person looks visibly anxious and nervous. "
                "Eyebrows slightly raised and pulled together, forehead tense with faint worry lines, "
                "eyes darting and tense, lips pressed together or slightly trembling, jaw slightly tight. "
                "Do not change anything else: background, lighting, clothing, hair, skin tone, face shape, camera angle."
            ),
            "neutral": (
                "Edit only the facial expression: the person has a completely neutral, relaxed expression. "
                "Face muscles fully relaxed, mouth closed with no tension, "
                "eyes open naturally and looking straight ahead, eyebrows in their natural resting position. "
                "Do not change anything else: background, lighting, clothing, hair, skin tone, face shape, camera angle."
            ),
            "confused": (
                "Edit only the facial expression: the person looks genuinely confused and puzzled. "
                "One eyebrow raised higher than the other, forehead slightly wrinkled on one side, "
                "eyes slightly squinted, mouth slightly asymmetric — one corner slightly pulled to the side. "
                "Do not change anything else: background, lighting, clothing, hair, skin tone, face shape, camera angle."
            ),
        }

    prompt = prompts.get(final_emotion, prompts["neutral"])
    try:
        image = replicate.run("black-forest-labs/flux-kontext-dev",
                input={
                        "input_image": image_code, 
                        "prompt": prompt,
                        "num_inference_steps": 50, 
                        "guidance": 3.5,
                        "output_format": "jpg"
                }
            )
            
        generated_image = str(image)
    except Exception as e:
        print(f"Error generating image: {e}")
        generated_image = None    
    
    

    return {"text": text, "dominant_emotion": final_emotion, "emotions_array": emotions_array, "generated_image": generated_image, "bot_response": bot_response}
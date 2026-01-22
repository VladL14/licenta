import torch
from torch.utils.data import Dataset
import csv 
import random

HAPPY_IDS = {1,4,13,15,17,18,20,21,23}
SAD_IDS = {9,16,24,25}
ANGRY_IDS = {2,3,10,11}
SURPRISED_IDS = {22,26}
ANXIOUS_IDS = {12,14,19}

class DatasetML(Dataset):
    def __init__(self, file_path, tokenizer, max_len=64, t_ratio=0.3):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                text = line[0]
                raw_id = line[1]
                label_id = line[1].split(",")
                label_ids = [int(id) for id in label_id if id.isdigit()]
                target = [0,0,0,0,0]
                has_emotion = False
                for id in label_ids:
                    if id in HAPPY_IDS:
                        target[0] = 1
                        has_emotion = True
                    if id in SAD_IDS:
                        target[1] = 1
                        has_emotion = True
                    if id in ANGRY_IDS:
                        target[2] = 1
                        has_emotion = True
                    if id in SURPRISED_IDS:
                        target[3] = 1
                        has_emotion = True
                    if id in ANXIOUS_IDS:
                        target[4] = 1
                        has_emotion = True
                if has_emotion :
                    self.data.append((text, target))
                else:
                    if random.random() < t_ratio:
                        self.data.append((text, target))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = str(self.data[index][0])
        target = self.data[index][1]


        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
        )

        return {
            'ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.float)
        }

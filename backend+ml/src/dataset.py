labels = ["happy", "sad", "angry", "surprised", "anxious"]
HAPPY_IDS = {1,4,13,15,17,18,20,21,23}
SAD_IDS = {9,16,24,25}
ANGRY_IDS = {2,3,10,11}
SURPRISED_IDS = {22,26}
ANXIOUS_IDS = {12,14,19}
UNMAPPED_SET = {0,5,6,7,8}
NEUTRAL_ID = 27

def parse_ids(labels):
    labels = labels.strip()
    if labels == "":
        return []
    return [int(id) for id in labels.split(",")]

def map_labels(label_ids):
    if NEUTRAL_ID in label_ids:
        return [0,0,0,0,0]
    label_vector = [0,0,0,0,0]
    for id in label_ids:
        if id in HAPPY_IDS:
            label_vector[0] = 1
        if id in SAD_IDS:
            label_vector[1] = 1
        if id in ANGRY_IDS:
            label_vector[2] = 1
        if id in SURPRISED_IDS:
            label_vector[3] = 1
        if id in ANXIOUS_IDS:
            label_vector[4] = 1
    return label_vector
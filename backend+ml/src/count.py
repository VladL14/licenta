labels = ["happy", "sad", "angry", "surprised", "anxious"]
HAPPY_IDS = {1,4,13,15,17,18,20,21,23}
SAD_IDS = {9,16,24,25}
ANGRY_IDS = {2,3,10,11}
SURPRISED_IDS = {22,26}
ANXIOUS_IDS = {12,14,19}
UNMAPPED_SET = {0,5,6,7,8}
NEUTRAL_ID = 27

count = {"happy":0, "sad":0, "angry":0, "surprised":0, "anxious":0}
all_zero = 0
total = 0
single_label = 0
multi_label = 0
neutral_count = 0
unmaped_count = 0

for line in open("./data/train.tsv", "r", encoding="utf-8"):
    total += 1
    parts = line.strip().split("\t")
    raw_id = parts[1]
    label_id = [int(id) for id in raw_id.split(",")]
    if label_id == [NEUTRAL_ID]:
        neutral_count += 1
    label_vector = [0,0,0,0,0]
    for id in label_id:
        if all(id in UNMAPPED_SET for id in label_id):
            unmaped_count += 1
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
    if sum(label_vector) == 0:
        all_zero += 1
    if sum(label_vector) == 1:
        single_label += 1
    if sum(label_vector) > 1:
        multi_label += 1
    if label_vector[0] == 1:
        count["happy"] += 1
    if label_vector[1] == 1:
        count["sad"] += 1
    if label_vector[2] == 1:
        count["angry"] += 1
    if label_vector[3] == 1:
        count["surprised"] += 1
    if label_vector[4] == 1:
        count["anxious"] += 1

print("Total samples:", total)
print("All zero samples:", all_zero)
for label in labels:
    print(f"{label} count: {count[label] /total*100:.2f}%")
print(f"all_zero percentage: {all_zero/total*100:.2f}%")
print(f"single_label samples: {single_label} percentage: {single_label/total*100:.2f}%")
print(f"multi_label samples: {multi_label} percentage: {multi_label/total*100:.2f}%")
print(f"neutral samples: {neutral_count} percentage: {neutral_count/total*100:.2f}%")
print(f"unmapped samples: {unmaped_count} percentage: {unmaped_count/total*100:.2f}%")


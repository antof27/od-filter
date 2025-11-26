import json

with open("/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/merged_dataset_from_pt.json", "r") as f:
    data = json.load(f)

count_0 = 0
count_1 = 0
for item in data:
    
    if item['cls'] == 0:
        count_0 += 1
    elif item['cls'] == 1:
        count_1 += 1

print(f"Label 0 count: {count_0}")
print(f"Label 1 count: {count_1}")

total = count_0 + count_1
print(total)
import json

with open("/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/train_set.json", "r") as f:
    json_data = json.load(f)

# count_0 = 0
# count_1 = 0
# for item in data:
    
#     if item['cls'] == 0:
#         count_0 += 1
#     elif item['cls'] == 1:
#         count_1 += 1

# print(f"Label 0 count: {count_0}")
# print(f"Label 1 count: {count_1}")

# total = count_0 + count_1
# print(total)


def count_images(input_data):
    # 1. If the input is a string, parse it.
    if isinstance(input_data, str):
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError as e:
            return f"Error reading JSON string: {e}"
            
    # 2. If the input is already a list, use it directly.
    elif isinstance(input_data, list):
        data = input_data
        
    else:
        return "Error: Input must be a JSON string or a Python list."

    unique_images = set()

    for item in data:
        full_id = item.get("id", "")
        if full_id:
            # Split by the last underscore and take the first part
            image_id = full_id.rsplit("_", 1)[0]
            unique_images.add(image_id)

    return len(unique_images)

# Run the function
count = count_images(json_data)
print(f"Total unique images found: {count}")
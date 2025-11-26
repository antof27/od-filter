import torch

data1 = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/yolo_predicted_embeddings/0a6ba0bd-d880-4a50-92e0-b1b3df278547.pt")
# data2 = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/concatenated_embeddings_with_normalized_bboxes/0a6ba0bd-d880-4a50-92e0-b1b3df278547.pt")
data2 = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/concat-embeds-1284/0a077a30-beec-460c-a77f-fa4196ecdc08.pt")

data3 = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/yolo_predicted_embeddings/72f95d60-cf26-4821-8d79-4ec72c748031.pt")

# data3 = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/gt_converted/0a6ba0bd-d880-4a50-92e0-b1b3df278547.pt")

# dino = torch.load("/storage/team/EgoTracksFull/v2/yolo-world-hooks/playground/dino_converted/0a6ba0bd-d880-4a50-92e0-b1b3df278547.pt")

# print(data1.keys())
print(data2.keys())

# print(data1["embeddings"][0].shape)
print(data2["embeddings"][0].shape)
print("data2 object:ids:", data2["object_ids"][:5])
# print("data2 bboxes:", data2["bboxes"][:5])

# len1 = len(data1["object_ids"])
# len2 = len(data2["object_ids"])

# total_length = len1 + len2

# print(f"Length of first data: {len1}")
# print(f"Length of second data: {len2}")
# print(f"Sum of lengths: {total_length}")

# print("n_objects, first 10:", data1["n_objects"][:10])

# #sum all the n_objects
# total_objects = sum(data1["n_objects"]) + sum(data2["n_objects"])
# print(f"Total number of objects in both datasets: {total_objects}")
# print(len(data1["object_ids"]))
# print(len(data1["embeddings"]))
# print(len(data1["bboxes"]))


# print the first 5 values of all the keys in data1 except embeddings
# for key in data1.keys():
#     if key != "embeddings":
#         print(f"{key}: {data1[key][:10]}")

# for key in data3.keys():
#         # if key!= "embeddings":
#             print(f"{key}: {data3[key][:10]}")


# print("DINO embeddings", dino.keys())
# print(dino["object_ids"][:10])
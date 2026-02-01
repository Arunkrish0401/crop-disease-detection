import os

DATASET_PATH = "data/PlantVillage"

classes = os.listdir(DATASET_PATH)

print("Number of disease classes:", len(classes))
print("Sample classes:")
for c in classes[:10]:
    print("-", c)

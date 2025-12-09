import os
import json

folders = os.listdir("/home/axie@SPE.local/MAUDE_summarizer/Datasets")

for folder in folders:
    #print(f"{folder}")
    keep = []
    with open(os.path.join("/home/axie@SPE.local/MAUDE_summarizer/Datasets", folder, "state.json"), "r") as file:
    # with open(f"Datasets/{folder}/state.json", "r") as file:
        data = json.load(file)
        for file in data["_data_files"]:
            keep.append(file["filename"])
        # print(keep)
    for file in os.listdir(os.path.join("/home/axie@SPE.local/MAUDE_summarizer/Datasets", folder)):
    # for file in os.listdir(f"Datasets/{folder}"):
        if file not in keep and ".arrow" in file:
            os.remove(os.path.join("/home/axie@SPE.local/MAUDE_summarizer/Datasets", folder, file))



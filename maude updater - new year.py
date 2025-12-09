# RUN THIS FILE TO CREATE A NEW DATASET FOR A NEW YEAR

import pandas as pd
import zipfile
import requests
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, Dataset, concatenate_datasets
import faiss
import datasets
import os
import torch

current_year = datetime.now().year

links = ["mdrfoi", "device", "foitext", "patient", "foidevproblem", "patientproblemcode"]

# download all files to be added to current year's dataset:
for link in links:
    url = f"https://www.accessdata.fda.gov/MAUDE/ftparea/{link}.zip"

    filename = f"{link}.zip"

    response = requests.get(url)

    with open(filename, "wb") as f:
        f.write(response.content)

        

    with zipfile.ZipFile(filename, "r") as zip_ref:
        for member in zip_ref.infolist():
            original_name = member.filename
            lower_name = original_name.lower()
            with zip_ref.open(member) as source, open(lower_name, "wb") as target:
                target.write(source.read())
dfs = {}
for link in tqdm(links):
    df = pd.read_csv(f"{link}.txt", sep="|", encoding_errors="ignore", engine="python", on_bad_lines="skip").fillna(" ")
    dfs[link] = df

# dataset cleaning and aggregation

dfs["mdrfoi"] = dfs["mdrfoi"][["MDR_REPORT_KEY", "ADVERSE_EVENT_FLAG", "REPORT_NUMBER", "DATE_OF_EVENT", "EVENT_TYPE"]]
print("MDR done!")

dfs["device"] = dfs["device"][["MDR_REPORT_KEY", "BRAND_NAME", "GENERIC_NAME", "MANUFACTURER_D_NAME", "MODEL_NUMBER", "DEVICE_REPORT_PRODUCT_CODE"]]
print("Device done!")

dfs["foitext"] = dfs["foitext"][["MDR_REPORT_KEY", "TEXT_TYPE_CODE", "FOI_TEXT"]]
#dfs["foitext"] = dfs["foitext"].drop_duplicates(subset="FOI_TEXT")
mask = dfs["foitext"]["TEXT_TYPE_CODE"] == "N"
dfs["foitext"] = pd.concat([
    dfs["foitext"][mask].drop_duplicates(subset="FOI_TEXT"),
    dfs["foitext"][~mask]
])

dfs["foitext"] = dfs["foitext"][~dfs["foitext"][['FOI_TEXT']].astype(str).apply(lambda row: row.str.contains (r'[a-z]').any(), axis=1)]
print("foitext done!")

dfs["patient"] = dfs["patient"][["MDR_REPORT_KEY", "SEQUENCE_NUMBER_OUTCOME"]]
print("Patient done!")

dfs["foidevproblem"] = dfs["foidevproblem"].rename(columns = {"619611":"MDR_REPORT_KEY", "1104": "DEVICE_PROBLEM_CODE"})
agg_devproblem = dfs["foidevproblem"].groupby("MDR_REPORT_KEY")["DEVICE_PROBLEM_CODE"].apply(lambda x: "; ".join(x.astype(str))).reset_index()
agg_devproblem["MDR_REPORT_KEY"] = agg_devproblem["MDR_REPORT_KEY"].astype(str)
dfs["foidevproblem"] = agg_devproblem
print("Device problem codes done!")

ppc = dfs["patientproblemcode"].drop(["PATIENT_SEQUENCE_NO", "DATE_CHANGED", "DATE_ADDED"], axis=1)
ppc["PROBLEM_CODE"] = pd.to_numeric(ppc["PROBLEM_CODE"], errors='coerce')
ppc = ppc.dropna(subset=["PROBLEM_CODE"])
ppc["PROBLEM_CODE"] = ppc["PROBLEM_CODE"].astype(int)
ppc = ppc.rename(columns = {"PROBLEM_CODE": "PATIENT_PROBLEM_CODE"}) 
agg_ppc= ppc.groupby("MDR_REPORT_KEY")["PATIENT_PROBLEM_CODE"].apply(lambda x: "; ".join(x.astype(str))).reset_index()
agg_ppc["MDR_REPORT_KEY"] = agg_ppc["MDR_REPORT_KEY"].astype(str)
dfs["patientproblemcode"] = ppc
print("Patient problem codes done!")

for key in tqdm(dfs.keys()):
    dfs[key]["MDR_REPORT_KEY"] = dfs[key]["MDR_REPORT_KEY"].astype(str)

combined = (
    dfs["foitext"]
    .groupby(["MDR_REPORT_KEY", "TEXT_TYPE_CODE"], as_index=False)
    .agg({"FOI_TEXT": lambda texts: "\n\n".join(texts)})
)

manu = combined[combined["TEXT_TYPE_CODE"] == "N"]
manu = manu.rename(columns = {"FOI_TEXT": "FOI_TEXT_MANU"})
manu = manu.drop("TEXT_TYPE_CODE", axis=1)

pt = combined[combined["TEXT_TYPE_CODE"] == "D"]
pt = pt.rename(columns = {"FOI_TEXT": "FOI_TEXT_PT"})
pt = pt.drop("TEXT_TYPE_CODE", axis=1)

test = pd.merge(manu,pt, on="MDR_REPORT_KEY", how="outer")
dfs["foitext"] = test
print("foi text split into event/manu!")


# dataset merging
merge1 = pd.merge(dfs["foitext"], dfs["device"], on="MDR_REPORT_KEY", how="left")
print("Merge 1 of 5 complete!")
merge2 = pd.merge(merge1, dfs["mdrfoi"], on="MDR_REPORT_KEY", how="left")
print("Merge 2 of 5 complete!")
merge3 = pd.merge(merge2, dfs["patientproblemcode"], how = "left", on="MDR_REPORT_KEY")
print("Merge 3 of 5 complete!")
merge4 = pd.merge(merge3, dfs["foidevproblem"], how="left", on="MDR_REPORT_KEY")
print("Merge 4 of 5 complete!")
merge_final = pd.merge(merge4, dfs["patient"], on="MDR_REPORT_KEY", how="left")
print("Merge 5 of 5 complete!")
merge_final.to_csv("testadd.csv")

keep_cols = ["MDR_REPORT_KEY", "FOI_TEXT_MANU", "FOI_TEXT_PT", "BRAND_NAME", "GENERIC_NAME", "MANUFACTURER_D_NAME", "MODEL_NUMBER", "DEVICE_REPORT_PRODUCT_CODE", "ADVERSE_EVENT_FLAG", "REPORT_NUMBER", "DATE_OF_EVENT", "EVENT_TYPE", "PATIENT_PROBLEM_CODE", "DEVICE_PROBLEM_CODE", "SEQUENCE_NUMBER_OUTCOME"]

# dropping duplicates and extra columns, and changing all columns into type string
_df = merge_final[keep_cols]
_df.drop_duplicates(subset="MDR_REPORT_KEY", keep="first")
for col in keep_cols:  
    if _df[col].dtype != "object":
        _df[col] = _df[col].astype(str)

column_order = ["MDR_REPORT_KEY", "FOI_TEXT_MANU", "FOI_TEXT_PT", "BRAND_NAME", "GENERIC_NAME", "MANUFACTURER_D_NAME",
                "MODEL_NUMBER", "DEVICE_REPORT_PRODUCT_CODE", "ADVERSE_EVENT_FLAG", "REPORT_NUMBER", 
                "DATE_OF_EVENT", "EVENT_TYPE", "PATIENT_PROBLEM_CODE", "DEVICE_PROBLEM_CODE", "SEQUENCE_NUMBER_OUTCOME"]

_df = _df[column_order]

def stringify(entry):
    st_builder = ""
    if pd.notna(entry["BRAND_NAME"]) and len(str(entry["BRAND_NAME"])) > 0:
        st_builder += f"Device brand name: {entry['BRAND_NAME']}\n"
    if pd.notna(entry["GENERIC_NAME"]) and len(str(entry["GENERIC_NAME"])) > 0:
        st_builder += f"Device generic name: {entry['GENERIC_NAME']}\n"
    if pd.notna(entry["MANUFACTURER_D_NAME"]) and len(str(entry["MANUFACTURER_D_NAME"])) > 0:
        st_builder += f"Manufacturer: {entry['MANUFACTURER_D_NAME']}\n"
    if pd.notna(entry["FOI_TEXT_PT"]) and len(str(entry["FOI_TEXT_PT"])) > 0:
        st_builder += f"Patient narrative: {entry['FOI_TEXT_PT']}\n"
    if pd.notna(entry["FOI_TEXT_MANU"]) and len(str(entry["FOI_TEXT_MANU"])) > 0:
        st_builder += f"Manufacturer narrative: {entry['FOI_TEXT_MANU']}\n"
    return st_builder
    
TXT = []
for _, row in tqdm(_df.iterrows(), total=len(_df)):
    TXT.append(stringify(row))
    
# creating embeddings and mapping them to dataset
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(TXT, show_progress_bar = True, normalize_embeddings = True, batch_size = 64)
del model
torch.cuda.empty_cache()

new_dataset = Dataset.from_pandas(_df, preserve_index=False)
del _df
assert(embeddings.shape[0] == new_dataset.shape[0])
new_column = [0] * len(embeddings)
new_dataset = new_dataset.add_column("trash", new_column)
new_dataset = new_dataset.map(
    lambda batch, idxs: {"embeddings": [embeddings[i] for i in idxs]},
    batched=True,
    with_indices=True,
    remove_columns=["trash"]
)

# save dataset and faiss indices
new_dataset.save_to_disk(f"Datasets/{current_year}_dataset")
new_dataset.add_faiss_index(column = "embeddings", index_name = "f2", metric_type = faiss.METRIC_INNER_PRODUCT, device = 0)
new_dataset.save_faiss_index("f2", f"Faiss Indices/{current_year}_faiss.faiss")

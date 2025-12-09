import pandas as pd
import zipfile
import requests
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, Dataset
import os
import faiss
import datasets
import torch

current_year = datetime.now().year
current_month = datetime.now().month

links = ["mdrfoichange", "devicechange", "foitextchange", "patientchange", "foidevproblem", "patientproblemcode"]
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
if current_year not in years:
    years.append(current_year)

# downloading all necessary files

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

# cleaning and aggregation of text files

dfs["mdrfoichange"] = dfs["mdrfoichange"][["MDR_REPORT_KEY", "ADVERSE_EVENT_FLAG", "REPORT_NUMBER", "DATE_OF_EVENT", "EVENT_TYPE"]]
print("MDR done!")

dfs["devicechange"] = dfs["devicechange"][["MDR_REPORT_KEY", "BRAND_NAME", "GENERIC_NAME", "MANUFACTURER_D_NAME", "MODEL_NUMBER", "DEVICE_REPORT_PRODUCT_CODE"]]
print("Device done!")

dfs["foitextchange"] = dfs["foitextchange"][["MDR_REPORT_KEY", "TEXT_TYPE_CODE", "FOI_TEXT"]]
#dfs["foitextchange"] = dfs["foitextchange"].drop_duplicates(subset="FOI_TEXT")
mask = dfs["foitextchange"]["TEXT_TYPE_CODE"] == "N"
dfs["foitextchange"] = pd.concat([
    dfs["foitextchange"][mask].drop_duplicates(subset="FOI_TEXT"),
    dfs["foitextchange"][~mask]
])

dfs["foitextchange"] = dfs["foitextchange"][~dfs["foitextchange"][['FOI_TEXT']].astype(str).apply(lambda row: row.str.contains (r'[a-z]').any(), axis=1)]
print("foitext done!")

dfs["patientchange"] = dfs["patientchange"][["MDR_REPORT_KEY", "SEQUENCE_NUMBER_OUTCOME"]]
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
    dfs["foitextchange"]
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
dfs["foitextchange"] = test
print("foi text split into event/manu!")

# merging into one dataframe

merge1 = pd.merge(dfs["foitextchange"], dfs["devicechange"], on="MDR_REPORT_KEY", how="left")
print("Merge 1 of 5 complete!")
merge2 = pd.merge(merge1, dfs["mdrfoichange"], on="MDR_REPORT_KEY", how="left")
print("Merge 2 of 5 complete!")
merge3 = pd.merge(merge2, dfs["patientproblemcode"], how = "left", on="MDR_REPORT_KEY")
print("Merge 3 of 5 complete!")
merge4 = pd.merge(merge3, dfs["foidevproblem"], how="left", on="MDR_REPORT_KEY")
print("Merge 4 of 5 complete!")
merge_final = pd.merge(merge4, dfs["patientchange"], on="MDR_REPORT_KEY", how="left")
print("Merge 5 of 5 complete!")
merge_final.to_csv("testchange.csv")

keep_cols = ["MDR_REPORT_KEY", "FOI_TEXT_MANU", "FOI_TEXT_PT", "BRAND_NAME", "GENERIC_NAME", "MANUFACTURER_D_NAME", "MODEL_NUMBER", "DEVICE_REPORT_PRODUCT_CODE", "ADVERSE_EVENT_FLAG", "REPORT_NUMBER", "DATE_OF_EVENT", "EVENT_TYPE", "PATIENT_PROBLEM_CODE", "DEVICE_PROBLEM_CODE", "SEQUENCE_NUMBER_OUTCOME"]

# dropping duplicates and extra columns, and changing all columns into type string

change_df = merge_final[keep_cols]
change_df = change_df.drop_duplicates(subset="MDR_REPORT_KEY", keep="first")
for col in keep_cols:  
    if change_df[col].dtype != "object":
        change_df[col] = change_df[col].astype(str)

column_order = ["MDR_REPORT_KEY", "FOI_TEXT_MANU", "FOI_TEXT_PT", "BRAND_NAME", "GENERIC_NAME", "MANUFACTURER_D_NAME",
                "MODEL_NUMBER", "DEVICE_REPORT_PRODUCT_CODE", "ADVERSE_EVENT_FLAG", "REPORT_NUMBER", 
                "DATE_OF_EVENT", "EVENT_TYPE", "PATIENT_PROBLEM_CODE", "DEVICE_PROBLEM_CODE", "SEQUENCE_NUMBER_OUTCOME"]

change_df = change_df[column_order]
        
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
for _, row in tqdm(change_df.iterrows(), total=len(change_df)):
    TXT.append(stringify(row))
    
# embedding texts    
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(TXT, show_progress_bar = True, normalize_embeddings = True, batch_size = 64)

del model
torch.cuda.empty_cache()

#creating new dataset and mapping embeddings to it
new_dataset = Dataset.from_pandas(change_df, preserve_index=False)
del change_df
assert(embeddings.shape[0] == new_dataset.shape[0])
new_column = [0] * len(embeddings)
new_dataset = new_dataset.add_column("trash", new_column)
new_dataset = new_dataset.map(
    lambda batch, idxs: {"embeddings": [embeddings[i] for i in idxs]},
    batched=True,
    with_indices=True,
    remove_columns=["trash"]
)
print("Change dataset created!")

# finding which years are actually in new_dataset to be updated to avoid excessive saving/loading
key_set = set(new_dataset["MDR_REPORT_KEY"]) 
all_years = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010]
if current_year not in all_years:
    all_years.insert(0, current_year)
years = []
for year in tqdm(all_years):
    ds = datasets.load_from_disk(f"Datasets/{year}_dataset")
    year_set = set(ds["MDR_REPORT_KEY"])
    for key2 in year_set:
        if key2 in key_set:
            years.append(year)
            break
print(years)


# replaces any "old" entries with "new" entries - based on MDR Report Key
# saves and overwrites dataset/faiss index
for year in tqdm(years):
    save_dir = os.path.join("Backups", f"{year}_backup")
    ds = datasets.load_from_disk(f"Datasets/{year}_dataset")
    ds.save_to_disk(save_dir)
    ds = datasets.load_from_disk(f"Backups/{year}_backup")
    assert(len(ds.column_names) == len(new_dataset.column_names))
    update_dict = {row["MDR_REPORT_KEY"]: row for row in new_dataset}
    def update_fn(example):
        return update_dict.get(example["MDR_REPORT_KEY"], example)
    ds_updated = ds.map(update_fn)
    ds_updated.save_to_disk(f"Datasets/{year}_dataset")
    ds_updated.add_faiss_index(column = "embeddings", index_name = "f2", 
    metric_type = faiss.METRIC_INNER_PRODUCT, device = 0)
    ds_updated.save_faiss_index("f2", f"Faiss Indices/{year}_faiss.faiss")
    print(f"{year} updated!")




    

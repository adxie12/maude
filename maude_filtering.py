import datasets
import faiss
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import re

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def filter1(df, year, search_type, brand, manu, model_num, code, must, must2, query, must_operator, must_contain, pattern):
    ds = datasets.load_from_disk(f"Datasets/{year}_dataset")
    # ds = DW.df_dict[year]

    ds_filtered = []

    if search_type and any([brand, manu, model_num, code, must]):
        ds_filtered = ds_filter(ds, manu, brand, model_num, code, must_contain, must_operator, must, must2, pattern)
        if len(ds_filtered) >= 1:
            ds_filtered.add_faiss_index(column="embeddings", index_name="f2", metric_type=faiss.METRIC_INNER_PRODUCT, device=0)
            scores, retrieved_examples = ds_filtered.get_nearest_examples("f2", model.encode(f"{query}"), k=2048)
        else:
            return None, 0, 0
        
    else:
        ds.load_faiss_index("f2", f"Faiss Indices/{year}_faiss.faiss")
        scores, retrieved_examples = ds.get_nearest_examples("f2", model.encode(f"{query}"), k=2048)
    inter = pd.DataFrame(retrieved_examples)
    inter["similarity_score"] = scores
    if search_type == False and must_contain:
        if must2:
            if must_operator == "OR":
                mask = inter.apply(lambda row: row.astype(str).str.contains(pattern, case=False, regex=True).any(), axis=1)
            else:
                mask = inter.apply(lambda r: r.astype(str).str.contains(must_contain[0], case=False, regex=True).any(), axis=1)
                for phrase in must_contain[1:]:
                    mask &= inter.apply(lambda r: r.astype(str).str.contains(phrase, case=False, regex=True).any(), axis=1)
        else:
            mask = inter.apply(lambda row: row.astype(str).str.contains(must, case=False, regex=True).any(), axis=1)
        inter_filt = inter[mask]
        # elif must:
        #     mask = inter.apply(lambda row: row.astype(str).str.contains(must, case=False, regex=True).any(), axis=1)
        #     inter_filt = inter[mask]
    else:
        inter_filt = inter
    
    df.append(inter_filt)


def prefiltered_filtering(df, stop_lock, stop_dict, search_id):
    df_filtered = df[df["FOI_TEXT_PT"].str.len() > 150]
    if df_filtered.shape[0] > 2048:
        df_filtered = df_filtered.iloc[:2048]
    duplicates = df_filtered.duplicated(subset="FOI_TEXT_PT", keep="first")
    dropped = df_filtered[duplicates]
    kept = df_filtered.drop_duplicates(subset="FOI_TEXT_PT", keep="first")

    kept["num_duplicates"] = 0
    for _, dropped_row in tqdm(dropped.iterrows(), total = dropped.shape[0]):
        with stop_lock:
            if stop_dict.get(search_id):
                print(f"Search {search_id} was cancelled.")
                stop_dict.pop(search_id)
                return None
        for idx, kept_row in kept.iterrows():
            if dropped_row["FOI_TEXT_PT"] == kept_row["FOI_TEXT_PT"]:
                kept.loc[idx, "REPORT_NUMBER"] = (
                    f"{kept_row['REPORT_NUMBER']} + {dropped_row['REPORT_NUMBER']}"
                )
                kept.loc[idx, "num_duplicates"] += 1

    return kept



def not_prefiltered_filtering(df, manu, brand, model_num, code, stop_lock, stop_dict, search_id):

    manu_mask, brand_mask, model_mask, code_mask = df_mask(df, manu, brand, model_num, code)

    df_filtered = df[manu_mask & brand_mask & model_mask & code_mask]  
    if df_filtered.shape[0] < 1:
        print("No results found")
        return df_filtered
    # if df_filtered.shape[0] <= 20:
    #     top_20 = df_filtered[:20]
    # else: 
    #     top_20 = df_filtered
    df_filtered_2 = prefiltered_filtering(df_filtered, stop_lock, stop_dict, search_id)
    return df_filtered_2


def ds_filter(ds, manu, brand, model_num, code, must_contain, must_operator, must, must2, pattern):

    def row_text(row):
        return ' '.join(str(v) for v in row.values())

    if must_contain:
        if must2:
            if must_operator == "OR":
                # OR: match if any phrase matches (combined pattern)
                ds = ds.filter(lambda row: bool(re.search(pattern, row_text(row), re.IGNORECASE)), num_proc = 14, batch_size = 10000)
            else:
                # AND: filter sequentially by each phrase
                for phrase in must_contain:
                    ds = ds.filter(lambda row, p=phrase: bool(re.search(p, row_text(row), re.IGNORECASE)), num_proc = 14, batch_size = 10000)
        else:
            # Single pattern from `must`
            ds = ds.filter(lambda row: bool(re.search(must, row_text(row), re.IGNORECASE)), num_proc = 14, batch_size = 10000)

    def normalize_input(val):
        if val is None or val == "":
            return []
        if isinstance(val, str):
            return val.split("+") if "+" in val else [val]
        if isinstance(val, list):
            return val
        return []

    manu = normalize_input(manu)
    brand = normalize_input(brand)
    model_num = normalize_input(model_num)
    code = normalize_input(code)

    manu = [m.lower().replace("'", "").strip() for m in manu]
    brand = [b.lower().replace("'", "").strip() for b in brand]
    model_num = [m.lower().strip() for m in model_num]
    code = [c.lower().strip() for c in code]

    def filter_fn(example):
        manu_val = (example.get("MANUFACTURER_D_NAME") or "").lower().replace("'", "").strip()
        brand_val = (example.get("BRAND_NAME") or "").lower().replace("'", "").strip()
        model_val = (example.get("MODEL_NUMBER") or "").lower().strip()
        code_val = (example.get("DEVICE_REPORT_PRODUCT_CODE") or "").lower().strip()

        return (
            (not manu or any(m in manu_val for m in manu)) and
            (not brand or any(b in brand_val for b in brand)) and
            (not model_num or any(m in model_val for m in model_num)) and
            (not code or any(c in code_val for c in code))
        )



    
    return ds.filter(filter_fn, num_proc = 14, batch_size = 10000)

def df_mask(df, manu, brand, model_num, code):
    if "+" in code:
        code = code.split("+")
    elif isinstance(code, str):
        code = [code]
    code_mask = pd.Series(False, index=df.index)
    for term in code:
        code_mask |= df["DEVICE_REPORT_PRODUCT_CODE"].str.lower().str.strip().str.contains(term.lower(), na=False)

    if "+" in manu:
        manu = manu.split("+")
    elif isinstance(manu, str):
        manu = [manu]
    manu_mask = pd.Series(False, index=df.index)
    for term in manu:
        manu_mask |= df["MANUFACTURER_D_NAME"].str.lower().str.replace("'", "", regex=False).str.strip().str.contains(term.lower().replace("'", ""), na = False)
    
    if "+" in brand:
        brand = brand.split("+")
    elif isinstance(brand,str):
        brand = [brand]
    brand_mask = pd.Series(False, index=df.index)
    for term in brand:
        brand_mask |= df["BRAND_NAME"].str.lower().str.replace("'", "", regex=False).str.strip().str.contains(term.lower().replace("'", ""), na=False)

    if "+" in model_num:
        model_num = model_num.split("+")
    elif isinstance(model_num, str):
        model_num = [model_num]
    model_mask = pd.Series(False, index=df.index)
    for term in model_num:
        model_mask |= df["MODEL_NUMBER"].str.lower().str.strip().str.contains(term.lower(), na=False)

    return manu_mask, brand_mask, model_mask, code_mask
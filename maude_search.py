import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import maude_filtering as MF
import maude_clustering as MC
from queue import Queue
import pickle

# for testing this file only - comment these lines out when running from main
# import LLMService as LM
# from threading import Lock
# service = LM.LLMService(LM.LLMServiceConfig)
# search_id = 1
# stop_dict = {}
# stop_lock = Lock()
# status_dict = {}

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
# cluster_dfs = {}
# cluster_dfs_relabelled = {}
# tokens = {}
# must_contain = []
# pattern = ""
# total = 0
# num_clusters = 0

def perform_search(search_id, service, query, brand, manu, model_num, code, must, must2, must_operator, search_type, years, stop_dict, stop_lock, status_dict):
    interim = {}
    if query.strip() == "":
        phrase = "Please enter a valid string to search!"
        return phrase
    else:
        # global must_contain
        must_contain = [phrase for phrase in [must, must2] if phrase]
        pattern = ""
        if must_operator == "OR":
            pattern = '|'.join(must_contain)

        df = []

        for year in years:
            with stop_lock:
                if stop_dict.get(search_id):
                    print(f"Search {search_id} was cancelled.")
                    stop_dict.pop(search_id)
                    return None, 0, 0
                status_dict[search_id] = f"Searching and filtering {year}..."
            MF.filter1(df, year, search_type, brand, manu, model_num, code, must, must2, query, must_operator, must_contain, pattern)

        if len(df) < 1:
            return None, 0, 0
        else:
            df = pd.concat(df, ignore_index=True)
        with stop_lock:
            if stop_dict.get(search_id):
                print(f"Search {search_id} was cancelled.")
                stop_dict.pop(search_id)
                return None, 0, 0
            status_dict[search_id] = "More filtering/getting rid of duplicates..."
        if search_type and len(df) > 0:
            df_filtered = MF.prefiltered_filtering(df, stop_lock, stop_dict, search_id)
        else:
            df_filtered = MF.not_prefiltered_filtering(df, manu, brand, model_num, code, stop_lock, stop_dict, search_id)
        if df_filtered is None:
            return None, 0, 0
        if df_filtered.shape[0] < 1:
            return None, 0, 0
        with stop_lock:
            if stop_dict.get(search_id):
                print(f"Search {search_id} was cancelled.")
                stop_dict.pop(search_id)
                return None, 0, 0
            status_dict[search_id] = "Clustering..."
        total = df_filtered.shape[0]
        if total == 0:
            print("No results found!")
            return None, 0, 0
        emb = df_filtered["embeddings"]
        emb = np.vstack(emb).astype('float32')
        df, total2 = MC.clustering(total, emb, df_filtered)
        with stop_lock:
            if stop_dict.get(search_id):
                print(f"Search {search_id} was cancelled.")
                stop_dict.pop(search_id)
                return None, 0, 0
        print(df.shape[0])
        labels = df["cluster"]
        len_clusters = len(np.unique(labels))
        print(len_clusters)
        if total > 0:
            cluster_dfs = {}

            for label in np.unique(labels):
                label_num = label + 1
                with stop_lock:
                    if stop_dict.get(search_id):
                        print(f"Search {search_id} was cancelled.")
                        stop_dict.pop(search_id)
                        return None, 0, 0
                    status_dict[search_id] = f"Breaking down cluster {label_num} of {len_clusters}..."
                cluster = df[df["cluster"] == label].drop("embeddings", axis=1, errors="ignore")
                split_clusters = MC.subcluster(str(label), cluster)
                cluster_dfs.update(split_clusters)

        # print(cluster_dfs.keys)
        cluster_dfs_relabelled = {}



        cluster_dfs_relabelled = {str(i+1): v for i, (_, v) in enumerate(cluster_dfs.items())}
        gen_text, clusters = summarize(cluster_dfs_relabelled, service, stop_lock, stop_dict, search_id, status_dict)
        interim["gen_text"] = gen_text
        if gen_text is None:
            return None, 0, 0
        status_dict[search_id] = "Creating summary table and CSV files..."
        contents = [entry['content'] for entry in gen_text]
        interim["contents"] = contents
        pickle.dump(interim, open("saved.pkl", "wb"))
        try:
            summary_df = summary_table(contents, query, clusters, total, search_id)
        except Exception as e:
            print(f"Caught {e} exception")
            pickle.dump(interim, open("saved.pkl", "wb"))
        summary_df.to_csv(f"maude output/{search_id}/!summary_{search_id}.csv")
        # global num_clusters
        num_clusters = len(cluster_dfs_relabelled.keys())


        return summary_df, num_clusters, total2

def summarize(clusters, service, stop_lock, stop_dict, search_id, status_dict):

    resultQ = Queue()


    # model_name = "Qwen/Qwen3-4B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model_summ = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="cuda"
    # )

    gen_text = []
    len_clusters = len(clusters)


    for label, df_cluster in tqdm(clusters.items()):
        with stop_lock:
            if stop_dict.get(search_id):
                print(f"Search {search_id} was cancelled.")
                stop_dict.pop(search_id)
                return None, None
            status_dict[search_id] = f"Summarizing cluster {label} of {len_clusters}..."
        cluster_rows_json = []

        for _, row in df_cluster.iterrows():
            row_dict = row.to_dict()
            row_json = json.dumps(row_dict, default = str)
            cluster_rows_json.append(row_json)


        prompt = f"""Each entry in this text is associated with cluster {label}. Summarize all of the patient and manufacturer narratives in this cluster in a few sentences. 
        Include the names of the devices and the manufacturers in each cluster. Return the summary in JSON format: {{Cluster: {label}, Event: ""}}. Do not return an empty string.
        
        {cluster_rows_json}
        """

        messages = [
            
            {"role": "user", "content": prompt},
        ]

        service.submit(messages, resultQ)
        service.processQueue()

    while not resultQ.empty():
        gen_text.append({"content": resultQ.get()})

    return gen_text, clusters



        
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize = False,
        #     add_generation_prompt = True,
        #     enable_thinking = False
        # )
        # model_inputs = tokenizer([text], return_tensors = "pt").to(model_summ.device)
        # input_len = model_inputs.input_ids.shape[1]
        # # print(input_len)

        # generated_ids = model_summ.generate(
        #     **model_inputs,
        #     max_new_tokens = 32768
        # )
        # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # try:
        #     index = len(output_ids) - output_ids[::-1].index(151668)
        # except ValueError:
        #     index = 0


        # content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        # gen_text.append({"content": content})

    # return gen_text

def summary_table(contents, query, clusters, total, search_id):
    entries = []
    if total > 1:
        for content in contents:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                content = match.group(0)
            # print(content)
            try:
                # content = content.replace("\\", "")
                data = json.loads(content)
                cluster = str(data.get("Cluster", "Unknown"))#.split("_")[0]
                event = data.get("Event", "No Event Summary")
                # data = json.loads(content.replace("\\", ""))
                # data = data.replace("\"", "")

                if cluster not in clusters:
                    print(f"Warning: Cluster '{cluster}' not in clusters dict.")
                    continue

            except json.JSONDecodeError as e:
                print("JSON error:", e)
                print("Offending content:\n", content)
                continue
            except UnboundLocalError as e:
                print("General error:", e)
                continue

            entries.append({
                "Cluster": cluster,
                "Cluster Summary": event,
                "Entries": clusters[cluster].shape[0]
            })

    else: 
        for content in contents:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                content = match.group(0)
            data = json.loads(content)
            cluster = str(data["Cluster"])#.split("_")[0]
            event = data["Event"]
            entries.append({
                "Cluster": cluster,
                "Cluster Summary": event,
                "Entries": 1
            })
    new_df = pd.DataFrame(entries)

    for key in clusters.keys():

        #move to appropriate function if time allows - not a very cohesive function if left here
        clusters[key].rename(columns={"MDR_REPORT_KEY": "MDR Report Key", "FOI_TEXT_MANU": "Manufacturer Narrative", "FOI_TEXT_PT": "Event Description",
        "BRAND_NAME": "Device Brand Name", "GENERIC_NAME": "Device Generic Name", "MANUFACTURER_D_NAME": "Manufacturer", "MODEL_NUMBER": "Model Number", 
        "DEVICE_REPORT_PRODUCT_CODE": "Product Code", "REPORT_NUMBER": "Report Number", "ADVERSE_EVENT_FLAG": "Adverse Event Happened?", "DATE_OF_EVENT": "Event Date", 
        "EVENT_TYPE": "Event Type",
        "SEQUENCE_NUMBER_OUTCOME": "Patient Outcome", "PATIENT_PROBLEM_CODE": "Patient Problem Code", "DEVICE_PROBLEM_CODE": "Device Problem Code", "similarity_score": "Similarity Score",
        "num_duplicates": "Number of Duplicates"}, inplace=True)
        # clusters[key].drop("cluster", axis=1, inplace=True)
        new_df["Entries"] = new_df["Entries"].astype(str)
        
        dupes = clusters[key]["Number of Duplicates"].sum()
        new_df.loc[new_df["Cluster"] == key, "Entries"] = (
        new_df.loc[new_df["Cluster"] == key, "Entries"].astype(str) + f" (+{dupes} duplicate(s))")
        mean = clusters[key]["Similarity Score"].mean()
        new_df.loc[new_df["Cluster"] == key, "Mean Similarity Score"] = mean
        prod_codes = clusters[key]["Product Code"]
        counts = prod_codes.value_counts()
        prod_codes = counts.index.tolist()
        # prod_codes = np.unique(cluster_dfs[key]["DEVICE_REPORT_PRODUCT_CODE"])
        prod_codes_str = ", ".join(prod_codes) 
        new_df.loc[new_df["Cluster"] == key, "Product Codes Contained"] = prod_codes_str
        
        df = clusters[key]
        df_M = df[df["Event Type"] == "M"]
        df_I = df[df["Event Type"] == "IN"]
        df_D = df[df["Event Type"] == "D"]
        df_O = df[~df["Event Type"].isin(["M", "IN", "D"])]


        I_total = df_I.shape[0]
        D_total = df_D.shape[0]
        M_total = df_M.shape[0]
        O_total = df_O.shape[0]

        new_df.loc[new_df["Cluster"] == key, "Event Types"] = f"M: {M_total} <br> I: {I_total} <br> D: {D_total} <br>O: {O_total} <br>"
        new_df_sorted = new_df.sort_values(by="Mean Similarity Score", ascending=False)

        # for label, df in clusters.items():
        clusters[key]["Link"] = clusters[key]["MDR Report Key"].apply(
            lambda x: f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfMAUDE/detail.cfm?mdrfoi__id={x}")
        if "cluster" in clusters[key].columns:
            clusters[key].drop("cluster", axis=1, inplace=True)

                

        output_dir = os.path.join("maude output", f"{search_id}")
        os.makedirs(output_dir, exist_ok = True)
        for _, row in new_df.iterrows():
            label = row["Cluster"]
            filename = os.path.join(output_dir, f"{query}_{label}.csv")
            clusters[label].to_csv(filename)
            # for item in cluster_dfs:
            #     item.to_csv(f"{}")
        
    return new_df_sorted



# if __name__ == "__main__":
#     perform_search(search_id, service, "wheelchair", "", "", "", "", "", "", "", False, [2025], stop_dict, stop_lock, status_dict)
    

# def perform_search(query, brand, manu, model_num, code, must, must2, must_operator, search_type, years):

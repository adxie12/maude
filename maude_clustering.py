from sklearn.cluster import AgglomerativeClustering
import numpy as np
from transformers import AutoTokenizer

total_samples = 0
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

def clustering(total, emb, df):
    print(f"Total results: {total}")
    if total > 1:
        agg = AgglomerativeClustering(distance_threshold=2, n_clusters=None)
        agg_cluster = agg.fit(emb)
        labels = agg_cluster.labels_
        df["cluster"] = labels
    else:
        labels = "0"
        df["cluster"] = labels

    samples = 500//len(np.unique(labels)) + 1
    if total > 500:
        test = df.sort_values(by=["cluster", "similarity_score"], ascending=[True, False])
        while True:
            top_per_cluster = test.groupby("cluster").head(samples).reset_index(drop=True)
            if top_per_cluster.shape[0] >= 500:
                break
            samples += 1
        global total_samples
        total_samples = top_per_cluster.shape[0]
        top_emb = top_per_cluster["embeddings"]
        top_emb = np.vstack(top_emb).astype('float32')
        emb = top_emb

        agg = AgglomerativeClustering(distance_threshold = 1 ,n_clusters=None)
        agg_cluster = agg.fit(top_emb)
        labels = agg_cluster.labels_
        labels += 1
        top_per_cluster["cluster"] = labels
        unique_labels = np.unique(labels)
        size = top_per_cluster.shape[0]
        print(f"# of entries: {size}")
        return top_per_cluster, size
    else:
        return df, total
    
def subcluster(label, cluster):
    cluster_str = cluster.to_string()
    encoded = tokenizer(cluster_str, return_tensors="pt")
    token_count = encoded.input_ids.shape[1]

    if token_count < 25000:
        return {label: cluster}
    
    # rare edge case - not sure how to handle
    # if token_count > 25000 and cluster.shape[0] == 1:

    
    if token_count > 25000 and cluster.shape[0] > 1:
        mid = cluster.shape[0] // 2
        first_half = cluster.iloc[:mid]
        second_half = cluster.iloc[mid:]

        result = {}
        result.update(subcluster(f"{label}_a", first_half))
        result.update(subcluster(f"{label}_b", second_half))
        return result
    return {label: cluster}

    # if token_count > 25000 and cluster.shape[0] > 1:
    #     mid = cluster.shape[0] // 2
    #     first_half = cluster.iloc[:mid]
    #     second_half = cluster.iloc[mid:]
    #     subcluster(f"{label}_a", first_half)
    #     subcluster(f"{label}_b", second_half)
    # else:
    #     MS.cluster_dfs[label] = cluster
    #     MS.tokens[label] = encoded
    #     print(f"Stored cluster {label} with {token_count} tokens and {cluster.shape[0]} rows.")

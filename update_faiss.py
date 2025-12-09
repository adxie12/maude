import datasets
import faiss
from datetime import datetime

year = datetime.now().year
ds = datasets.load_from_disk(f"/home/axie@SPE.local/MAUDE_summarizer/Datasets/{year}_dataset")
index = faiss.read_index(f"/home/axie@SPE.local/MAUDE_summarizer/Faiss Indices/{year}_faiss.faiss")
if len(ds) != index.ntotal:
    ds.add_faiss_index(column = "embeddings", index_name = "f2", metric_type = faiss.METRIC_INNER_PRODUCT, device = 0)
    ds.save_faiss_index("f2", f"/home/axie@SPE.local/MAUDE_summarizer/Faiss Indices/{year}_faiss.faiss")
	    

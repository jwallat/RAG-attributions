# %%
import torch

print("Start loading docs")
split_docs = torch.load("/home/wallat/RAG/data/faiss/wiki/split_docs.pt")

print("Docs loaded\n\n")
print(split_docs[0].page_content)


# %%
len(split_docs)

# %%
# Write jsonlines file of the documents
import json
from tqdm import tqdm

f = None
for i, doc in tqdm(enumerate(split_docs)):
    if i % 100000 == 0:
        if f is not None:
            f.close()
        f = open(f"/home/wallat/RAG/data/faiss/wiki/split_docs_{i}.jsonl", "w")
    f.write(json.dumps({"id": f"doc{i}", "contents": doc.page_content, "metadata": doc.metadata}) + "\n")

# with open("/home/wallat/RAG/data/faiss/wiki/pyserini/split_docs.jsonl", "w") as f:
#     for i, doc in enumerate(split_docs):
#         # print(doc)
#         # item = {"id": f"doc{i}", "contents": doc.page_content}
#         f.write(json.dumps({"id": f"doc{i}", "contents": doc.page_content, "metadata": doc.metadata}) + "\n")

# %%
# !python -m pyserini.index.lucene  --collection JsonCollection \  
#    --input /home/wallat/RAG/data/faiss/wiki/pyserini/ \
#    --index sample_collection_jsonl \
#    --generator DefaultLuceneDocumentGenerator \
#    --threads 1 \
#    --storeRaw

# %%
# from pyserini.search.lucene import LuceneSearcher

# searcher = LuceneSearcher('/home/wallat/bm25_kilt_wiki/sample_collection_jsonl')
# hits = searcher.search('first letter')

# for i in range(len(hits)):
#     print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')
#     full_doc = searcher.doc(hits[i].docid)
#     print(full_doc.raw())

    # print(hits[0].keys())



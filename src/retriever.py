import json
from tqdm import tqdm
import subprocess
from pyserini.search.lucene import LuceneSearcher


class Retriever():

    def __init__(self, config):
        self.config = config

    def retrieve(self, query):
        # Retrieve data from database
        return ...


class FaissRetriever(Retriever):

    def __init__(self, config):
        super().__init__(config)

    def create_index(self):
        # Prepare index
        return ...

    def embedd_queries(self, queries):
        # Embedd queries
        return ...

    def retrieve(self, query):
        # Retrieve data from Faiss database
        return ...
    

class BM25Retriever(Retriever):

    def __init__(self, config):
        super().__init__(config)

        # self.retriever = None
        self.retriever = LuceneSearcher('/home/wallat/RAG/data/faiss/wiki/pyserini/wikipedia-kilt')


    def convert_docs_to_pyserini_format(self, docs, path):
        f = None
        for i, doc in tqdm(enumerate(docs)):
            if i % 500000 == 0:
                if f is not None:
                    f.close()
                f = open(f"{path}/split_docs_{i}.jsonl", "w")
            f.write(json.dumps({"id": f"doc{i}", "contents": doc.page_content, "metadata": doc.metadata}) + "\n")


    def run_index_process(self, out_folder, json_folder, num_threads=10):
        command = [
            'python3', '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection',
            '--input', json_folder,
            '--index', f"{out_folder}/wikipedia-kilt",
            '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', str(num_threads),
            '--storeRaw'
            # '--storePositions', '--storeDocvectors', '--storeRaw'
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


    def create_index(self, split_docs, store_path="/home/wallat/RAG/data/faiss/wiki/pyserini/"):
        # Prepare index
        self.convert_docs_to_pyserini_format(split_docs, store_path)
        self.run_index_process(store_path, store_path)
    

    def retrieve(self, query, k):
        # Retrieve data using BM25
        hits = self.retriever.search(query, k=k)

        retrieved_docs = []

        for i in range(len(hits)):
            # print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')
            full_doc = self.retriever.doc(hits[i].docid)
            doc_content = json.loads(full_doc.raw())['contents']
            retrieved_docs.append(doc_content)
        
        return retrieved_docs
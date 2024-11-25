from tqdm import tqdm
from typing import Optional, List
import datasets
import wandb
import time
import json
from langchain.docstore.document import Document as LangchainDocument
from ragatouille import RAGPretrainedModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline

import logging
import hydra
from omegaconf import DictConfig

from src.retriever import BM25Retriever, FaissRetriever, Retriever
from src.llm import data_generator, get_reader_LLM
from src.models import LLMReader
from src.reranker import get_reranker
from src.vector_store import get_vector_store
import pandas as pd


log = logging.getLogger(__name__)



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    log.info("Starting up...")

    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Run dir: {run_dir}")

    # Do not sync the wandb run if we are running on the cluster
    wandb.init(project=cfg.wandb.project_name, config=dict(cfg), dir=run_dir, mode=cfg.wandb.mode)

    
    # Load the dataset

    # CR+_old
    # df = pd.read_json("/home/wallat/RAG/post_ration_random_data.jsonl", lines=True)
    # df = pd.read_json("/home/wallat/RAG/post_ration_rel_uncited_data.jsonl", lines=True)
    # df = pd.read_json("/home/wallat/RAG/post_ration_cited_other_data.jsonl", lines=True)

    # CR+_new
    df = pd.read_json(cfg.dataset.post_rationalization_ds, lines=True)
    # df = pd.read_json("/home/wallat/RAG/data/predictions/CR+_new/post_ration_rel_uncited_data.jsonl", lines=True)
    # df = pd.read_json("/home/wallat/RAG/data/predictions/CR+_new/post_ration_cited_other_data.jsonl", lines=True)



    dataset = []
    for row in df.iterrows():
        row = row[1]
        question = row['question']
        docs = row['selected_docs']
        citation_before = row['citation_before']
        adversarial_id = row['adversarial_id']
        dataset.append({'question': question, 'selected_docs': docs, 'citation_before': citation_before, 'adversarial_id': adversarial_id})


    llm = get_reader_LLM(cfg)

    data_gen = data_generator(llm, dataset)

    llm_pipeline = llm.load_model_pipeline()

    index = 0
    with tqdm(total=len(dataset), desc="LLM inference") as t:
        for pipe_outs in llm_pipeline(data_gen):
            for out in pipe_outs:
                answer = out["generated_text"]

                dataset[index]["answer"] = answer

                t.update(1)
                index += 1

    # Save the results to file
    with open(f"{run_dir}/results.jsonl", "w") as f:
        # f.write("[")
        for item in dataset:
            f.write(json.dumps(item) + "\n")
        # f.write("]")





if __name__ == "__main__":
    main()

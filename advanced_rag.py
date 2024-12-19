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


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    log.info("Starting up...")

    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Run dir: {run_dir}")

    # Do not sync the wandb run if we are running on the cluster
    wandb.init(
        project=cfg.wandb.project_name,
        config=dict(cfg),
        dir=run_dir,
        mode=cfg.wandb.mode,
    )

    if cfg.retriever.type == "faiss":
        retriever = FaissRetriever(cfg)
        # vector_store, embedding_model = get_vector_store(cfg)
    elif cfg.retriever.type == "bm25":
        retriever = BM25Retriever(cfg)
    reranker = get_reranker(cfg)
    llm = get_reader_LLM(cfg)

    # Get the task dataset
    task_ds = get_QA_dataset(cfg.dataset.name)

    process_task_ds(
        task_ds,
        llm,
        run_dir,
        retriever,
        reranker,
    )


def get_QA_dataset(ds_name):
    task_ds = []

    if ds_name == "wiki":
        dataset = datasets.load_dataset("kilt_tasks", name="nq", split="test")
        for row in dataset:
            task_ds.append(
                {
                    "id": row["id"],
                    "question": row["input"],
                    "answer": row["output"],
                }
            )

    elif ds_name == "squad":
        dataset = datasets.load_dataset("rajpurkar/squad_v2", split="validation[:2]")
        for row in dataset:
            task_ds.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gold_answer": row["answers"],
                }
            )

    return task_ds


def process_task_ds(
    dataset: List[dict],
    llm: LLMReader,
    run_dir: str,
    retriever: Retriever,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
):

    if False:
        # Faiss Retriever
        # TODO: needs to be put into the retriever class
        log.info("=> Embedd queries...")
        start_time = time.time()

        questions = [row["question"] for row in dataset]
        question_embeddings = embedding_model.embed_documents(questions)
        end_time = time.time()
        log.info(f"Embedding queries took: {end_time - start_time} seconds")

        log.info("=> Retrieving documents...")
        i = 0
        for row in tqdm(dataset, desc="Retrieving documents"):
            retrieved_docs = knowledge_index.similarity_search_by_vector(
                question_embeddings[i], k=num_retrieved_docs
            )
            retrieved_docs_content = [
                doc.page_content for doc in retrieved_docs
            ]  # Keep only the text
            row["retrieved_docs"] = retrieved_docs_content
            i += 1

    log.info("=> Retrieving documents...")

    questions = [row["question"] for row in dataset]

    i = 0
    for row in tqdm(dataset, desc="Retrieving documents"):
        retrieved_docs = retriever.retrieve(questions[i], k=num_retrieved_docs)
        row["retrieved_docs"] = retrieved_docs
        i += 1

    if reranker:
        log.info("=> Reranking documents...")
        for row in tqdm(dataset, desc="Reranking documents"):
            reranked_docs = reranker.rerank(
                row["question"], row["retrieved_docs"], k=num_docs_final
            )
            reranked_docs_content = [doc["content"] for doc in reranked_docs]

            row["selected_docs"] = reranked_docs_content[:num_docs_final]
    else:
        for row in tqdm(dataset):
            row["selected_docs"] = row["retrieved_docs"][:num_docs_final]

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

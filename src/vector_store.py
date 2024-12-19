from typing import Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import torch
import logging
from omegaconf import DictConfig
import time

from src.dataset import prepare_data


log = logging.getLogger(__name__)


def get_vector_store(cfg: DictConfig) -> Tuple[FAISS, HuggingFaceEmbeddings]:

    embedding_model = get_embedding_model(cfg.retriever.embedding_model_name)

    # Try loading any stored vectore DBs
    if cfg.retriever.use_cached_embeddings:
        try:
            start_time = time.time()
            vector_store = FAISS.load_local(
                f"/home/wallat/RAG/data/faiss/{cfg.dataset.name}/",
                embedding_model,
                allow_dangerous_deserialization=True,
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            log.info(f"Loading faiss index took: {elapsed_time} seconds")

            return vector_store, embedding_model
        except:
            log.info("No vector database found, creating a new one")

            # docs_processed = prepare_data(cfg)
            # vector_store, _ = get_document_store(cfg, docs_processed)

    docs_processed = prepare_data(cfg)
    vector_store, _ = get_document_store(cfg, docs_processed)

    return vector_store, embedding_model


def get_embedding_model(embedding_model_name: str) -> HuggingFaceEmbeddings:
    log.info("Building embedding model: ")

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=False,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,
        },  # Set `True` for cosine similarity
        show_progress=True,
    )

    return embedding_model


def get_document_store(cfg, docs_processed) -> FAISS:
    embedding_model = get_embedding_model(
        embedding_model_name=cfg.retriever.embedding_model_name
    )
    log.info("Starting to ingest documents into FAISS database...")

    vector_store = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    log.info("Done ingesting documents into FAISS database")

    vector_store.save_local(f"/home/wallat/RAG/data/faiss/{cfg.dataset.name}/")

    return vector_store

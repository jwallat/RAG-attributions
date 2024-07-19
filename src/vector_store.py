from langchain_text_splitters import TokenTextSplitter
from tqdm import tqdm
from typing import Optional, List, Tuple
import datasets

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from transformers import AutoTokenizer
from transformers import AutoTokenizer

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

import torch
from transformers import AutoTokenizer

import logging
from omegaconf import DictConfig
import time

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

def prepare_data(cfg: DictConfig) -> List[LangchainDocument]:

    if not cfg.dataset.use_cached_dataset:

        log.info("Preapring data:")

        if cfg.dataset.name == "wiki":
            dataset = datasets.load_dataset("kilt_wikipedia", split="full")
            log.info(f"Dataset has so many items: {len(dataset)}")

            RAW_KNOWLEDGE_BASE = [
                LangchainDocument(
                    page_content="".join(line for line in doc["text"]["paragraph"]),
                    metadata={
                        "anchors": doc["anchors"],
                        "wikidata_info": doc["wikidata_info"],
                        "kilt_id": doc["kilt_id"],
                        "wikipedia_id": doc["wikipedia_id"],
                        "wikipedia_title": doc["wikipedia_title"],
                    },
                )
                for doc in tqdm(dataset)
            ]

        elif cfg.dataset.name == "squad":
            # Squad v2
            # dataset = datasets.load_dataset("rajpurkar/squad_v2", split="train")
            dataset = datasets.load_dataset("rajpurkar/squad_v2", split="train[:10]")
            RAW_KNOWLEDGE_BASE = [
                LangchainDocument(
                    page_content=doc["context"], metadata={"source": doc["id"]}
                )
                for doc in tqdm(dataset)
            ]

        del dataset
        log.info(f"So many docs in KB: {len(RAW_KNOWLEDGE_BASE)}")

        torch.save(
            RAW_KNOWLEDGE_BASE, f"/home/wallat/RAG/data/faiss/{cfg.dataset.name}/RAW.pt"
        )

    else:
        log.info("Started loading the preprocessed documents. This may take a while...")
        RAW_KNOWLEDGE_BASE = torch.load(
            f"/home/wallat/RAG/data/faiss/{cfg.dataset.name}/RAW.pt"
        )

    log.info("Before splitting my documents")

    docs_processed = split_my_documents(
        100,  # We choose a chunk size adapted to our model
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=cfg.retriever.embedding_model_name,
    )

    return docs_processed


def split_my_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: str,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    # text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    #     AutoTokenizer.from_pretrained(tokenizer_name),
    #     chunk_size=chunk_size,
    #     chunk_overlap=0,
    #     add_start_index=True,
    #     strip_whitespace=True,
    #     # separators=MARKDOWN_SEPARATORS,
    # )
    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True),
        chunk_size=chunk_size,
        chunk_overlap=0,
        add_start_index=True,
        strip_whitespace=True,
    )

    log.info("Splitting documents using the hierarchical character splitter...")
    docs_processed = []
    for doc in tqdm(knowledge_base):
        # log.info(f"\n\n\n\nHere is the full doc: {doc.page_content}")
        split_docs = text_splitter.split_documents([doc])

        # log.info(f"\nThere are {len(split_docs)} split docs")
        # log.info(f"\n\nHere is the split doc: {split_docs[0]}")

        for split_doc in split_docs:
            if "wikipedia_title" in doc.metadata:
                split_doc.page_content = (
                    doc.metadata["wikipedia_title"] + "\n" + split_doc.page_content
                )
            # log.info(f"\n\nHere is a split doc: {split_doc.page_content}")
            docs_processed.append(split_doc)

    log.info("Done splitting documents.")

    log.info("Removing duplicates...")

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in tqdm(docs_processed):
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    log.info("Done removing duplicates.")

    log.info("\n\nHere are 2 example documents:")
    log.info(docs_processed_unique[0])
    log.info(docs_processed_unique[42])

    return docs_processed_unique


def get_embedding_model(embedding_model_name: str) -> HuggingFaceEmbeddings:
    log.info("Building embedding model: ")

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # Set `True` for cosine similarity
        show_progress=True,
    )

    return embedding_model


def get_document_store(cfg, docs_processed) -> FAISS:
    embedding_model = get_embedding_model(embedding_model_name=cfg.retriever.embedding_model_name)
    log.info("Starting to ingest documents into FAISS database...")

    vector_store = FAISS.from_documents(docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE)
    log.info("Done ingesting documents into FAISS database")

    vector_store.save_local(f"/home/wallat/RAG/data/faiss/{cfg.dataset.name}/")

    return vector_store
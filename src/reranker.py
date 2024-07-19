from omegaconf import DictConfig
from ragatouille import RAGPretrainedModel


def get_reranker(cfg: DictConfig):

    reranker = RAGPretrainedModel.from_pretrained(cfg.reranker.model_name)

    return reranker

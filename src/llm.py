from typing import List
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from src.models import CommandRPlusModel, LLMReader, Llama3Model


log = logging.getLogger(__name__)


def get_reader_LLM(cfg: DictConfig) -> LLMReader:
    model_name = cfg.model.name

    # TODO: All magic numbers should go into the model confic file    
    if "command-r-plus" in model_name:
        model = CommandRPlusModel(cfg)
    
    elif "llama-3" in model_name:
        model = Llama3Model(cfg)


    return model
    

def data_generator(model: LLMReader, dataset: List[dict]):
    for row in dataset:
        # Build the final prompt
        # context = "\n\nExtracted documents:\n"
        # context += "".join(
        #     [
        #         f"Document {str(i)}:::\n" + doc
        #         for i, doc in enumerate(row["selected_docs"])
        #     ]
        # )

        # final_prompt = prompt_template.format(question=row["question"], context=context)
        # log.info(f"Final prompt:\n\n{final_prompt}")

        final_prompt = model.get_input_in_model_format(row["question"], row["selected_docs"])

        yield final_prompt


def get_default_template(tokenizer):
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Cite the source document when relevant with for example [2] for document 2.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]
    prompt_template = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    log.info("RAG prompt template:")
    log.info(prompt_template)

    return prompt_template
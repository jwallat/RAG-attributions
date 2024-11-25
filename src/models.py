from abc import ABC
from typing import List
from omegaconf import DictConfig
import torch
import transformers
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class LLMReader(ABC):

    def __init__(self, cfg: DictConfig):
        # self.pipeline = self.load_model_pipeline(cfg)
        self.cfg = cfg


    def load_model_pipeline(self, cfg: DictConfig) -> transformers.pipeline:
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )

        pipe = build_pipe(cfg, self.tokenizer)
        return pipe

    def get_input_in_model_format(self, question: str, documents: List[str]) -> str:
        # TODO: standard prompt should also come from the config file
        default_prompt = [
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
        return self.tokenizer.apply_chat_template(
            default_prompt, tokenize=False, add_generation_prompt=True
        )
        

def build_pipe(cfg, tokenizer):

    generation_config = {"max_new_tokens": cfg.model.max_new_tokens}

    pipe = transformers.pipeline(
        "text-generation",
        model=cfg.model.name,
        tokenizer=tokenizer,
        device="cuda",
        torch_dtype=torch.bfloat16,
        batch_size=cfg.model.batch_size,
        trust_remote_code=True,
        return_full_text=False,
        **generation_config,
    )
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    return pipe


class Llama3Model(LLMReader):

    def load_model_pipeline(self) -> transformers.pipeline:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )

        pipe = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            device="cuda",
            torch_dtype=torch.bfloat16,
            batch_size=self.cfg.model.batch_size,
            trust_remote_code=True,
            **self.cfg.model.generation_config,
        )
        pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
        pipe.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        return pipe

    def get_input_in_model_format(self, system_prompt, question):
        # Standard huggingface chat template works
        return LLMReader.get_input_in_model_format(self, system_prompt, question)

    def handle_response(self, generated_text):
        return generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[
            -1
        ].strip()



class CommandRPlusModel(LLMReader):

    def load_model_pipeline(self) -> transformers.pipeline:
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_name,
        #     padding_side="left",
        #     use_fast=True,
        #     trust_remote_code=True,
        # )

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)

        model = AutoModelForCausalLM.from_pretrained(self.cfg.model.name, quantization_config=quant_config, trust_remote_code=True)

        pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            batch_size=self.cfg.model.batch_size,
            trust_remote_code=True,
            return_full_text=False,
            **self.cfg.model.generation_config,
        )
        # pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

        return pipe

    def get_input_in_model_format(self, question: str, documents: List[str]) -> str:
        conversation = [
            {"role": "user", "content": question}
        ]

        docs = []
        for doc in documents:
            docs.append({"text": doc})

        print("Question: ", question)
        print("Documents: ", docs)

        # render the tool use prompt as a string:
        grounded_generation_prompt = self.tokenizer.apply_grounded_generation_template(
            conversation,
            documents=docs,
            citation_mode="accurate", # or "fast"
            tokenize=False,
            add_generation_prompt=True,
            
        )

        return grounded_generation_prompt
# %%
question = "Whats the biggest penguin in the world?"

documents = [
    {
        "title": "Tall penguins",
        "text": "Emperor penguins are the tallest growing up to 122 cm in height.",
    },
    {"title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica."},
]

# %% [markdown]
# ### CommandR+

# %%
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import pipeline


# model_id = "CohereForAI/c4ai-command-r-plus"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# pipe = pipeline("text-generation", model=model_id)

# # define conversation input:
# conversation = [{"role": "user", "content": question}]

# # render the tool use prompt as a string:
# grounded_generation_prompt = tokenizer.apply_grounded_generation_template(
#     conversation,
#     documents=documents,
#     citation_mode="accurate",  # or "fast"
#     tokenize=False,
#     add_generation_prompt=True,
# )
# print(grounded_generation_prompt)


# %%
# output = pipe(grounded_generation_prompt)

# %%
# print(output)

# %% [markdown]
# ### General Stategies for grounded generation

# %% [markdown]
# #### Tool Calling

# %% [markdown]
# #### Direct Prompting

# %%
xml_system = """You're a helpful AI assistant. Given a user question and some Wikipedia article snippets, \
answer the user question and provide citations. If none of the articles answer the question, just say you don't know.

Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that \
justifies the answer and the ID of the quote article. Return a citation for every quote across all articles \
that justify the answer. Use the following format for your final output:

<response>
    <answer></answer>
    <citations>
        <citation><source_id></source_id><quote></quote></citation>
        <citation><source_id></source_id><quote></quote></citation>
        ...
    </citations>
</response>

Here are the Wikipedia articles:{context}"""

# %%
from typing import List
from langchain_core.output_parsers import XMLOutputParser


def format_docs_xml(docs) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        doc_str = f"""\
    <source id=\"{i}\">
        <title>{doc['title']}</title>
        <article_snippet>{doc['text']}</article_snippet>
    </source>"""
        formatted.append(doc_str)
    return "\n\n<sources>" + "\n".join(formatted) + "</sources>"


system_prompt = xml_system.format(context=format_docs_xml(documents))

# %%


# %%
import transformers
import torch

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "princeton-nlp/gemma-2-9b-it-SimPO"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# %%
outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    return_full_text=False,
)
print(outputs)


from langchain_core.output_parsers import XMLOutputParser

parser = XMLOutputParser(
    tags=["answer", "citation", "citations", "source_id", "Tall penguins"]
)

xml_out = outputs[0]["generated_text"]
print(parser.invoke(xml_out))

# %%
# txt = [{'generated_text': '<Tall penguins>\n    <answer>Emperor penguins</answer>\n    <citations>\n        <citation><source_id>0</source_id>The tallest penguins are Emperor penguins, growing up to 122 cm in height.</citation>\n    </citations>\n</Tall penguins>'}]
# txt = [
#     {
#         "generated_text": "<Tallpenguins>\n    <answer>Emperor penguins</answer>\n    <citations>\n        <citation><source_id>0</source_id>The tallest penguins are Emperor penguins, growing up to 122 cm in height.</citation>\n    </citations>\n</Tallpenguins>"
#     }
# ]
# txt = [
#     {
#         "generated_text": "<answer>\n    The biggest penguin in the world is the Emperor penguin.\n    <citations>\n        <citation><source_id>0</source_id>Emperor penguins are the tallest growing up to 122 cm in height.</citation>\n    </citations>\n</answer>"
#     }
# ]


# print(txt[0]["generated_text"])

# xml_out = txt[0]["generated_text"]
# print(parser.invoke(xml_out))


# %%
# xml_out = txt[0]["generated_text"].replace("\n", "")
# xml_out = xml_out.replace(" ", "")
# # xml_out = xml_out.strip()

# x = f"""{xml_out}"""
# x

# parser.invoke(x)


# %%
# import re
# import xml.etree.ElementTree as ET


# def remove_whitespace_between_tags(xml_string):
#     # Regex pattern to match and remove whitespace between XML tags
#     pattern = r'>\s+<'
#     cleaned_xml = re.sub(pattern, '><', xml_string)
#     return ET.fromstring(cleaned_xml)

# # Example XML string with whitespace between elements

# xml_out = remove_whitespace_between_tags(xml_out)
# xml_out

# %%
# from langchain_core.output_parsers import XMLOutputParser

# parser = XMLOutputParser(tags=["answer", "citation", "citations", "source_id", "Tall penguins"])

# print(parser.invoke(xml_out))

# %% [markdown]
# #### LLM Post-Processing

# %%
# pip install 'transformers>=4.39.1'
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

model_id = "CohereForAI/c4ai-command-r-plus-08-2024"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config)

# Format message with the command-r-plus-08-2024 chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
input_ids = input_ids.to(model.device)
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)


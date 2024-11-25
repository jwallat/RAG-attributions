# %%
import json

predictions_file_path = (
    "/home/wallat/RAG/command_r_attributions_natural_questions.jsonl"
)

with open(predictions_file_path, "r") as f:
    predictions = [json.loads(line) for line in f]

print(predictions[0].keys())

# %%
predictions[2]["answer"]

# %%
# get all invidual citations for each claim

grounded_answer = predictions[2]["answer"].split("Grounded answer: ")[1]
grounded_answer

# %%
import regex as re


def extract_citations(grounded_answer):

    pattern = r"<co: ([\d,]+)>(.*?)<\/co: \1>"
    matches = re.findall(pattern, grounded_answer)
    citations = []

    for match in matches:
        documents = match[0]
        claim = match[1]
        citations.append({"claim": claim, "documents": documents.split(",")})
        # print(f'Claim: "{claim}", provenance document: {documents}')

    return citations


citations = extract_citations(grounded_answer)
citations

# %%
import nltk
from typing import List

nltk.download("punkt")


# %%
def split_into_sentences(text) -> List[str]:
    # Step 1: Remove the tags using regex
    text = re.sub(r"<co: [\d,]+>(.*?)<\/co: [\d,]+>", r"\1", text)

    # Step 2: Tokenize the cleaned text into sentences
    sentences = nltk.sent_tokenize(text)
    return sentences


sentences = split_into_sentences(grounded_answer)

for i, sentence in enumerate(sentences):
    print(f"Sentence {i+1}: {sentence}")


# %%
def add_source_sentence_to_citations(grounded_answer, citations):

    sentences = split_into_sentences(grounded_answer)

    for citation in citations:
        # Match claim back to one of the sentences
        # print(f"Looking for claim '{citation['claim']}' in the sentences")
        selected_sentence_id = -1
        for i, sentence in enumerate(sentences):
            # print("Current sentence: ", sentence)
            if citation["claim"] in sentence:
                citation["sentence"] = sentence
                selected_sentence_id = i
                break
        if selected_sentence_id > 0:
            del sentences[:i]
        elif selected_sentence_id == -1:
            raise Exception(
                f"Claim '{citation['claim']}' not found in any of the sentences"
            )
        # print(citation)
    return citations


citations = add_source_sentence_to_citations(grounded_answer, citations)

# %%
num_attributions = 0
num_correct_citations = 0
wrong_citations = []
all_citations = []

for prediction in predictions:
    print("\n\n\nQuestion:", prediction["question"])

    try:
        grounded_answer = prediction["answer"].split("Grounded answer: ")[1]
        print("Grounded answer:", grounded_answer)
        # print("\n")

        citations = extract_citations(grounded_answer)
        citations = add_source_sentence_to_citations(grounded_answer, citations)

        for citation in citations:
            print(f"\nSource sentence: {citation['sentence']}")
            print(f"Claim: '{citation['claim']}'")
            print(f"Cited documents: {citation['documents']}")
            print("---")
            for attribution in citation["documents"]:
                num_attributions += 1
                print(
                    f"Cited document {attribution}: \n{prediction['selected_docs'][int(attribution)]}"
                )
                print("---")
                # if citation['claim'] in prediction['selected_docs'][int(attribution)]:
                #     num_correct_citations += 1
                # else:
                #     wrong_citations.append({'claim': citation['claim'], 'document': prediction['selected_docs'][int(attribution)], 'sentence': citation['sentence']})
                all_citations.append(
                    {
                        "claim": citation["claim"],
                        "document": prediction["selected_docs"][int(attribution)],
                        "sentence": citation["sentence"],
                    }
                )
    except Exception as e:
        print("\n\n\nThere was an error processing this QA pair")
        print(e)
        print("\n\n")
        print(prediction["answer"])


# %%
print(f"Number of attributions: {num_attributions}")
print(f"Number of correct citations: {num_correct_citations}")
print(f"Number of wrong citations: {num_attributions - num_correct_citations}")
print(f"Accuracy: {num_correct_citations / num_attributions}")

# %%
wrong_citations[:2]

# %% [markdown]
# ### Metrics

# %%
# import wandb
from evaluate import load

# from src.bem_pt import init_bem_model, predict_bem
from tqdm import tqdm
import string


def split_answers_with_multiple_options(data, model_predictions, question_type):
    dataset = []

    for index in range(0, len(data)):
        row = data.iloc[index]

        try:
            question = row["Question"]
            answers = str(row["Answer"])
        except:
            # No questions in dataset -> task is event ordering or fact checking, so we take the
            # model input as question
            question = model_predictions.iloc[index]["input"]
            answers = str(model_predictions.iloc[index]["ground_truth"])

        # answers = row["Answer"]
        if "__or__" in answers:
            answers = answers.split("__or__")
        else:
            answers = [answers]

        dataset.append(
            {
                "id": str(index),
                "question": question,
                "answers": answers,
                "type": question_type,
            }
        )

    return dataset


def convert_to_references(dataset):
    # Convert to reference format (evaluate library)
    # {'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '0'}
    references = []

    for ele in dataset:
        answers = ele["answers"]
        ref = {
            "id": str(ele["id"]),
            "answers": {"answer_start": len(answers) * [0], "text": answers},
        }

        references.append(ref)

    return references


def extract_questions(dataset):
    questions = []

    for ele in dataset:
        questions.append(ele["question"])

    return questions


def convert_to_predictions(model_predictions):
    predictions = []

    for index in range(0, len(model_predictions)):
        row = model_predictions.iloc[index]

        predictions.append({"prediction_text": row["answer"], "id": str(row["q_id"])})

    # predictions[14]
    return predictions


def contains(ref, pred):
    ref = ref.lower()
    pred = pred.lower()

    ref = ref.translate(str.maketrans("", "", string.punctuation))
    pred = pred.translate(str.maketrans("", "", string.punctuation))

    return ref in pred


def compute_contains_metric(references, predictions):
    # New metric "contains answer"
    num_contains = 0
    scores = []

    print(
        "*********************************************************** Computing contains metric ***********************************************************"
    )

    for pred, ref in zip(predictions, references):
        print("\n\nprediction: ", pred["prediction_text"])
        print("reference: ", ref["answers"]["text"])

        try:
            predicted_answer = pred["prediction_text"].lower()
        except:
            # print(pred["prediction_text"])
            # Special character
            print("Skipped contains computation")
            continue
        ref_answers = ref["answers"]["text"]
        ref_answers = [x.lower() for x in ref_answers]
        ref_answers = [
            x.translate(str.maketrans("", "", string.punctuation)) for x in ref_answers
        ]

        predicted_answer = predicted_answer.translate(
            str.maketrans("", "", string.punctuation)
        )

        print("\n\nprediction processed: ", predicted_answer)
        print("reference processed: ", ref_answers)

        contained = False
        for ref_answer in ref_answers:
            if ref_answer in predicted_answer:
                contained = True

        if contained:
            num_contains += 1
            scores.append(1)
        else:
            scores.append(0)

        print("contained: ", contained)

    return (num_contains / len(references)) * 100, scores


def compute_bemscore(references, predictions, questions):
    # The default in the paper https://arxiv.org/pdf/2202.07654.pdf​ is 0.5 - Minimal improvement
    # has been observed when directly tuning the treshold to 0.56 on the training set.
    threshold = 0.5
    preds = []
    for pred in predictions:
        preds.append(pred["prediction_text"])

    refs = []
    for ref in references:
        refs.append(ref["answers"]["text"])

    bem, tokenizer = init_bem_model()

    scores = []
    for question, references, prediction in tqdm(zip(questions, refs, preds)):

        match_score = 0
        for reference in references:
            if predict_bem(bem, question, reference, prediction, tokenizer) > threshold:
                match_score = 1

        scores.append(match_score)
    bem_score = sum(scores) / len(scores)

    return bem_score, scores


# def compute_openeval(references, predictions, questions):
#     preds = []
#     for pred in predictions:
#         preds.append(pred["prediction_text"])

#     refs = []
#     for ref in references:
#         refs.append(ref["answers"]["text"])

#     openeval_pipe = init_openeval_pipeline()

#     scores = []
#     for question, references, prediction in tqdm(zip(questions, refs, preds)):

#         match_score = 0
#         for reference in references:
#             if predict_openeval(openeval_pipe, question, reference, prediction) == 1:
#                 match_score = 1
#         scores.append(match_score)

#     openeval_score = sum(scores) / len(scores)

#     return openeval_score, scores


def compute_metrics(
    args,
    references,
    predictions,
    questions,
    path,
    single_metric=None,
    model_predictions=None,
):
    if single_metric == "squad":
        squad_metric = load("omidf/squad_precision_recall")
        return squad_metric.compute(predictions=predictions, references=references)
    elif single_metric == "contains":
        return compute_contains_metric(references, predictions)
    # elif single_metric == "OE":
    #     return compute_openeval(references, predictions, questions)
    elif single_metric == "BEM":
        return compute_bemscore(references, predictions, questions)

    else:
        # Compute all metrics
        squad_metric = load("omidf/squad_precision_recall")
        results = squad_metric.compute(predictions=predictions, references=references)
        results["contains"], contains_scores = compute_contains_metric(
            references, predictions
        )

        # results["OE_score"], oe_scores = compute_openeval(references, predictions, questions)
        results["BEM_score"], bem_scores = compute_bemscore(
            references, predictions, questions
        )

        if model_predictions is not None:
            model_predictions["contains"] = contains_scores
            # model_predictions["OE_score"] = oe_scores
            model_predictions["BEM_score"] = bem_scores

            model_predictions.to_csv(path, index=False, sep="\t")

    return results


def evaluate(
    args, data, model_predictions, ds_name, question_type, path=None, single_metric=None
):
    dataset = split_answers_with_multiple_options(
        data, model_predictions, question_type
    )

    references = convert_to_references(dataset)
    predictions = convert_to_predictions(model_predictions)
    questions = extract_questions(dataset)

    results = compute_metrics(
        args, references, predictions, questions, path, single_metric, model_predictions
    )

    # for key in results.keys():
    #     wandb.log({f"{ds_name} {question_type} {key}": results[key]})
    # wandb.log(results)

    return results


# %%
all_citations[:2]

# %%
contains_vals = []
# f1_vals = []
recall_vals = []

squad_metric = load("omidf/squad_precision_recall")

for citation in tqdm(all_citations):
    ref = citation["claim"]
    pred = citation["document"]
    contains_vals.append(contains(ref, pred))

    ref = [citation["claim"]]
    pred = citation["document"]
    results = squad_metric.compute(
        predictions=[{"prediction_text": pred, "id": "0"}],
        references=[{"id": "0", "answers": {"answer_start": [0], "text": ref}}],
    )
    # f1_vals.append(results['f1'])
    recall_vals.append(results["recall"])

print(f"Contains: {sum(contains_vals) / len(contains_vals)}")
# print(f"F1: {sum(f1_vals) / len(f1_vals)}")
print(f"Recall: {sum(recall_vals) / len(recall_vals)}")

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)

premise = "Blind carbon copy\nBlind carbon copy\nBlind carbon copy (abbreviated Bcc:) allows the sender of a message to conceal the person entered in the Bcc: field from the other recipients. This concept originally applied to paper correspondence and now also applies to email.\nIn some circumstances, the typist creating a paper correspondence must ensure that multiple recipients of such a document do not see the names of other recipients. To achieve this, the typist can: \nBULLET::::-'"
hypothesis = "BCC stands for blind carbon copy."


def predict_NLI(premise, hypothesis):

    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to("cuda"))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {
        name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)
    }
    # print(prediction)

    return prediction


NLI_vals = []

for citation in tqdm(all_citations):
    print("\n")
    print(citation["sentence"])
    print(citation["claim"])
    print(citation["document"])
    # NLI_vals.append(predict_NLI(citation["claim"], citation["document"]))
    NLI_vals.append(predict_NLI(citation["sentence"], citation["document"]))


# %%
NLI_vals

# %%
nli_vals = []

for val in NLI_vals:
    # print(val)
    judgement = max(val, key=val.get)
    if judgement == "entailment":
        nli_vals.append(1)
    elif judgement == "neutral":
        nli_vals.append(0)
    else:
        nli_vals.append(0)
    # print(judgement)
    # break

print(f"NLI sentence + doc: {sum(nli_vals) / len(nli_vals)}")

# %%
# Two analysis options:
# 1. Do NLI only for sentences that have only 1 claim since cheking the entire sentence with multiple claims agains one document might not be enough
# 2. Do NLI for each sentence and append all documents that were used in claims for that sentence


# TODO: Reorder the citations array so that all citations for the same sentence are together

citations_by_sentence = {}
# group citations by sentence
for citation in all_citations:
    sentence = citation["sentence"]
    if sentence in citations_by_sentence.keys():
        citations_by_sentence[sentence].append(citation)
    else:
        citations_by_sentence[sentence] = [citation]

citations_by_sentence.keys()


# for i, citation in tqdm(enumerate(citations)):
#     if

# %%
# NLI only for sentences with 1 claim

# TODO: This is not perfect because currently there is a 1:1 mapping between claims and docs. So if a claim cites 2 docs, the there will be 2 citations with claim, doc1 and claim, doc2.

NLI_vals = []

for sentence, citations in tqdm(citations_by_sentence.items()):
    if len(citations) == 1:
        citation = citations[0]
        # print("\n")
        # print(sentence)
        # print(citation['claim'])
        # print(citation['document'])
        NLI_vals.append(predict_NLI(sentence, citation["document"]))
        # NLI_vals.append(predict_NLI(citation['sentence'], citation['document']))

nli_vals = []

for val in NLI_vals:
    # print(val)
    judgement = max(val, key=val.get)
    if judgement == "entailment":
        nli_vals.append(1)
    elif judgement == "neutral":
        nli_vals.append(0)
    else:
        nli_vals.append(0)
    # print(judgement)
    # break

print(f"NLI sentences with only 1 claim: {sum(nli_vals) / len(nli_vals)}")

# %%
# 2. Do NLI for each sentence and append all documents that were used in claims for that sentence


NLI_vals = []
# i = 0
for sentence, citations in tqdm(citations_by_sentence.items()):
    # i += 1
    # if i > 100:
    #     break
    # print("\n")
    # print(sentence)
    # print(citation['claim'])
    # print(citation['document'])
    NLI_vals.append(
        predict_NLI(
            sentence, " ".join([citation["document"] for citation in citations])
        )
    )

nli_vals = []

for val in NLI_vals:
    # print(val)
    judgement = max(val, key=val.get)
    if judgement == "entailment":
        nli_vals.append(1)
    elif judgement == "neutral":
        nli_vals.append(0)
    else:
        nli_vals.append(0)
    # print(judgement)
    # break

print(f"NLI sentence with all cited documents: {sum(nli_vals) / len(nli_vals)}")

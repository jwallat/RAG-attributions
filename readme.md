# RAG Attributions

Goal:
"Certify" attributions -> guarantee correctness

Subgoals:

- Measure how many attributions are currently wrong
- Understand whether models produce attributions in a post-hoc manner by looking for retrieved documents that best match internal belives
- Test to show that the document is actually the reason for an attribution (=perturbations)

## Setup

You will want to run this on a big GPU, we are currently using [Command-R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit), which even in 4bit quantization requires a >40GB GPU.

### Datasets

We use the [KILT wikipedia snapshot](https://huggingface.co/datasets/facebook/kilt_wikipedia), which has corresponding open-domain QA datasets - of which we currently use [NaturalQuestions](https://huggingface.co/datasets/facebook/kilt_tasks). In both cases, we load them using the huggingface datasets package. So no manual downloading required ;)

### Environment

Setup conda and install requirements

```bash
conda create -n RAG python=3.10
conda activate RAG
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Running Experiments

You may acquire the attributed answers by running

```bash
python3 advanced_rag.py ++dataset.name="wiki" ++model.generation_config.max_new_tokens=512 ++model.name="CohereForAI/c4ai-command-r-plus-08-2024" ++model.batch_size=1
```

This will produce a hydra output folder containing logs and the models predictions. The first run will create an index (per default a pyserini BM25 index over the wikipedia snapshot).

### Faithful Citations Results

After acquiring the standard predictions, we inject adversarial statements into the LLM's context. We create 3 new dataset using the script in `attribution_analysis.ipynb` (in the section "Post-Rationalization Test Dataset"). Running the first 3 cells of that section will result in 3 datasets

- post_ration_random_data.jsonl
- post_ration_rel_uncited_data.jsonl
- post_ration_cited_other_data.jsonl

We will again run the model inference for these 3 files. This time using the `qnd.py` script as follows:

```bash
python3 qnd.py ++dataset.post_rationalization_ds="/home/wallat/RAG/data/predictions/CR+_2024/post_ration_random_data.jsonl" ++model.generation_config.max_new_tokens=512 ++model.name="CohereForAI/c4ai-command-r-plus-08-2024" ++model.batch_size=1
```

swapping the `++dataset.post_rationalization_ds=<file>` with our 3 files.

Lastly, we move back to the "Post-Rationalization Test *Eval*" section in `attribution_analysis.ipynb` and run the cell with our 3 new prediction files acquired from the earlier step.

## Slurm and Execution Tips

### Interactive mode

```bash
srun -t 30:00 --gpus=0 --cpus-per-task=4 /home/wallat/.conda/envs/RAG/bin/python advanced_rag.py -i
```

### Change hydra config

This project uses hydra to manage the configurations. You can find the config files in the `/conf` folder. If you want to overwrite the default values, you can use the following notation:

```bash
python advanced_rag.py ++dataset.name="squad"
```

### Quick test runs

Since working on the entire wikipedia corpus takes quite some time, you may switch the dataset.name config to "squad", which then uses the squad dataset instead of wikipedia.

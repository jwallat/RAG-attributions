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

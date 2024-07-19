# RAG Party


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


run slurm interactive mode:
srun -t 30:00 --gpus=0 --cpus-per-task=4 /home/wallat/.conda/envs/RAG/bin/python advanced_rag.py ++dataset.name="squad" -i
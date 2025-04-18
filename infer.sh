# BSUB -o ./bjob_logs/train_physion_4.5.%J

# BSUB -q gpu-compute

# BSUB -gpu "num=2:mode=shared:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB" 
# BSUB -J diffsynth-studio


source ~/.bashrc
conda activate diffusion-pipe
cd /project/osprey/scratch/x.zhexiao/video_gen/DiffSynth-Studio

# python infer_lora.py
torchrun --standalone --nproc_per_node=2 infer_lora_multi.py
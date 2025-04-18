import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image

# 下载模型
# snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="models/Wan-AI/Wan2.1-I2V-14B-480P")

# 加载模型
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32,
)
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,
)


# 下载示例图片
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern="data/examples/wan/input_image.jpg"
)
image = Image.open("data/examples/wan/input_image.jpg")

# 加载 LoRA 权重
# lora_weight_path = "/project/osprey/scratch/x.zhexiao/video_gen/diffusion-pipe/output/diffusion_pipe_training_runs/wan_480_test/20250324_19-11-26/epoch3/adapter_model.safetensors"
lora_weight_path = "output/lightning_logs/version_2/checkpoints/epoch=19-step=12000.ckpt"
# pipe.unet.load_attn_procs(lora_weight_path, alpha=1.0)  # 根据需要调整 alpha 参数
model_manager.load_lora(lora_weight_path, lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)
# pipe.enable_vram_management(num_persistent_param_in_dit=None)
# 进行 Image-to-video 推理
video = pipe(
    prompt="A boat move around from left to the right.",
    negative_prompt="",
    input_image=image,
    num_inference_steps=50,
    seed=0,
    tiled=True
)
save_video(video, "video.mp4", fps=15, quality=5)





# import torch
# from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData


# model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
# model_manager.load_models([
#     "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
#     "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
#     "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
# ])
# model_manager.load_lora("models/lightning_logs/version_1/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
# pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
# pipe.enable_vram_management(num_persistent_param_in_dit=None)

# video = pipe(
#     prompt="...",
#     negative_prompt="...",
#     num_inference_steps=50,
#     seed=0, tiled=True
# )
# save_video(video, "video.mp4", fps=30, quality=5)
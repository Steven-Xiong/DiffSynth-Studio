import torch
import os
import matplotlib.pyplot as plt
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image

# 定义一个全局字典，用于存储捕获的 attention 权重
attention_maps = {}

# 定义 hook 函数
def save_attention_hook(module, input, output):
    # 这里假设 output 为一个元组，第二项为 attention 权重（请根据实际情况调整）
    if isinstance(output, (list, tuple)) and len(output) > 1:
        attn = output[1]
        # 保存当前模块的 attention 权重，注意这里将权重移动到 CPU 并 detach
        key = module.__class__.__name__ + "_" + str(id(module))
        attention_maps[key] = attn.detach().cpu()
        print(f"Hook triggered: {key}, attn shape: {attn.shape}")

def register_attention_hooks(model):
    # 遍历模型中的所有子模块，并在包含 "attn" 字样或为 MultiheadAttention 的模块上注册 hook
    for name, module in model.named_modules():
        if "attn" in name.lower() or isinstance(module, torch.nn.MultiheadAttention):
            module.register_forward_hook(save_attention_hook)

def save_all_attention_maps(save_dir="attention_maps", prefix="i2v", step=0):
    os.makedirs(save_dir, exist_ok=True)
    for key, attn_map in attention_maps.items():
        # 假设 attn_map shape 为 (batch, num_heads, seq_len, seq_len)
        # 取第一个样本，并对所有 head 求平均
        avg_attn = attn_map[0].mean(dim=0).numpy()  # shape: (seq_len, seq_len)
        plt.figure(figsize=(6, 6))
        plt.imshow(avg_attn, cmap="viridis")
        plt.title(f"{prefix}_{key}_step{step}")
        plt.colorbar()
        save_path = os.path.join(save_dir, f"{prefix}_{key}_step{step}.png")
        plt.savefig(save_path)
        plt.close()
    # 清空字典，避免下次重复累积
    attention_maps.clear()

# 下载模型
# snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="models/Wan-AI/Wan2.1-I2V-14B-480P")

# 加载模型
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32,  # 图像编码器使用 float32
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
    torch_dtype=torch.bfloat16,  # 可设置为 torch.float8_e4m3fn 启用 FP8 量化
)

pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)  # 根据需求调整 VRAM 参数

# 在推理前，注册 attention hook 到去噪模型中
register_attention_hooks(pipe.denoising_model())

# 下载示例图片
from modelscope import dataset_snapshot_download
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=f"data/examples/wan/input_image.jpg"
)
image = Image.open("data/examples/wan/input_image.jpg")

# Image-to-video 推理
video = pipe(
    prompt="",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=image,
    num_inference_steps=50,
    seed=0, tiled=True
)

# 推理后，保存注意力图（例如保存一次推理的 attention）
save_all_attention_maps(save_dir="attention_maps", prefix="i2v", step=0)

# 保存生成的视频
save_video(video, "video.mp4", fps=15, quality=5)

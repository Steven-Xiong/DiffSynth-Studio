import os
import glob
import base64
import torch
import torch.distributed as dist
from PIL import Image
from diffsynth import ModelManager, WanVideoPipeline, save_video
from modelscope import snapshot_download, dataset_snapshot_download
from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
import matplotlib.pyplot as plt

# ---------------------------
# 分布式环境初始化
# ---------------------------
dist.init_process_group(backend="nccl", init_method="env://")
init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
initialize_model_parallel(
    sequence_parallel_degree=dist.get_world_size(),
    ring_degree=1,
    ulysses_degree=dist.get_world_size(),
)
torch.cuda.set_device(dist.get_rank())
rank = dist.get_rank()
world_size = dist.get_world_size()

# ---------------------------
# 辅助函数定义
# ---------------------------
def export_to_video(video, dest, fps=24):
    save_video(video, dest, fps=fps, quality=5)

def encode_video_to_data_url(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
    encoded = base64.b64encode(video_data).decode('utf-8')
    return f"data:video/mp4;base64,{encoded}"

# 定义全局字典存储 attention 权重
attention_maps = {}

# Hook 函数：捕获 attention 层输出（假设返回 (output, attn_weights)）
def save_attention_hook(module, input, output):
    if isinstance(output, (list, tuple)) and len(output) > 1:
        attn = output[1]
        key = module.__class__.__name__ + "_" + str(id(module))
        attention_maps[key] = attn.detach().cpu()
        print(f"Hook triggered: {key}, attn shape: {attn.shape}")

# 遍历模型子模块，在符合条件的 attention 层注册 hook
def register_attention_hooks(model):
    for name, module in model.named_modules():
        if "attn" in name.lower() or isinstance(module, torch.nn.MultiheadAttention):
            module.register_forward_hook(save_attention_hook)

# 保存当前捕获的 attention 权重为图片，保存后清空字典，防止累积
def save_all_attention_maps(save_dir, prefix, step):
    os.makedirs(save_dir, exist_ok=True)
    import pdb; pdb.set_trace()
    for key, attn_map in attention_maps.items():
        # 假设 attn_map 的 shape 为 (batch, num_heads, seq_len, seq_len)
        avg_attn = attn_map[0].mean(dim=0).numpy()  # 对所有 head 求平均，取第一个 batch
        plt.figure(figsize=(6, 6))
        plt.imshow(avg_attn, cmap="viridis")
        plt.title(f"{prefix}_{key}_step{step}")
        plt.colorbar()
        save_path = os.path.join(save_dir, f"{prefix}_{key}_step{step}.png")
        plt.savefig(save_path)
        plt.close()
    attention_maps.clear()

# ---------------------------
# 模型加载及 pipeline 初始化
# ---------------------------
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

# 加载 LoRA 权重（根据需要修改权重路径）
lora_weight_path = "output/lightning_logs/version_2/checkpoints/epoch=12-step=7800.ckpt"
model_manager.load_lora(lora_weight_path, lora_alpha=1.0)

# 创建 pipeline，多 GPU 环境下启用 USP 模式
pipe = WanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device=f"cuda:{rank}",
    use_usp=True if world_size > 1 else False
)
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# 注册 attention hook 到去噪模型
register_attention_hooks(pipe.denoising_model())

# ---------------------------
# 推理文件夹和输出文件夹配置
# ---------------------------
input_folder = "/project/osprey/scratch/x.zhexiao/video_gen/DiffSynth-Studio/data/example_physionv3_short"
output_folder = "/project/osprey/scratch/x.zhexiao/video_gen/DiffSynth-Studio/output/physionv3_short_4.5_7800"
if rank == 0:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
dist.barrier()

# ---------------------------
# 多 GPU 推理：处理本地属于当前 rank 的图片
# ---------------------------
all_image_paths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))
local_html_entries = []
for idx, image_path in enumerate(all_image_paths):
    if idx % world_size != rank:
        continue
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(input_folder, base_name + ".txt")
    
    if not os.path.exists(txt_path):
        print(f"Rank {rank}: Warning: 未找到 {txt_path}，跳过 {image_path}")
        continue

    with open(txt_path, "r", encoding="utf-8") as f:
        full_prompt = f.read().strip()

    image = Image.open(image_path)
    
    # 进行 Image-to-Video 推理，同时触发前向传播中的 attention hook
    video = pipe(
        prompt=full_prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=image,
        num_inference_steps=50,
        seed=0,
        tiled=True
    )
    
    # 保存生成的视频
    dest = os.path.join(output_folder, base_name + "_generated.mp4")
    export_to_video(video, dest, fps=24)
    
    # 保存当前推理过程中的 attention 图（存放到对应 rank 的文件夹下）
    attn_save_dir = os.path.join(output_folder, f"attention_maps_rank{rank}")
    save_all_attention_maps(save_dir=attn_save_dir, prefix=base_name, step=idx)
    
    # 构造 HTML 预览（生成 base64 data URL）
    gen_video_data_url = encode_video_to_data_url(dest)
    gt_video_path = image_path.rsplit(".", 1)[0] + ".mp4"
    if os.path.exists(gt_video_path):
        gt_video_data_url = encode_video_to_data_url(gt_video_path)
    else:
        gt_video_data_url = ""
    
    entry_html = f'''
    <div style="display: flex; align-items: flex-start; margin-bottom: 20px;">
        <div style="flex: 1; margin-right: 20px;">
            <h3>Caption</h3>
            <p>{full_prompt}</p>
        </div>
        <div style="flex: 1; margin-right: 20px;">
            <h3>Generated Video</h3>
            <video width="320" height="240" controls>
                <source src="{gen_video_data_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        <div style="flex: 1;">
            <h3>Ground-Truth Video</h3>
            <video width="320" height="240" controls>
                <source src="{gt_video_data_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
    </div>
    '''
    local_html_entries.append(entry_html)
    print(f"Rank {rank}: 已处理 {image_path}")

# ---------------------------
# 收集 HTML 结果并生成汇总页面（仅 rank 0 执行）
# ---------------------------
if rank == 0:
    all_html_entries = [None] * world_size
else:
    all_html_entries = None
dist.gather_object(local_html_entries, all_html_entries, dst=0)

if rank == 0:
    merged_entries = []
    for sublist in all_html_entries:
        if sublist:
            merged_entries.extend(sublist)
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Video Generation Results</title>
    </head>
    <body>
        {"".join(merged_entries)}
    </body>
    </html>
    '''
    html_path = os.path.join(output_folder, "results.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Rank {rank}: HTML 结果已保存至 {html_path}")

dist.barrier()
dist.destroy_process_group()

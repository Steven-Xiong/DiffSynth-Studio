import os
import glob
import base64
import torch
import torch.distributed as dist
from PIL import Image
from diffsynth import ModelManager, WanVideoPipeline, save_video
from modelscope import snapshot_download, dataset_snapshot_download

# 用于分布式初始化的辅助函数（确保已安装 xfuser 库）
from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment

# 初始化分布式环境
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

# 定义辅助函数
def export_to_video(video, dest, fps=24):
    save_video(video, dest, fps=fps, quality=5)

def encode_video_to_data_url(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
    encoded = base64.b64encode(video_data).decode('utf-8')
    return f"data:video/mp4;base64,{encoded}"

# 初始化并加载模型
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
lora_weight_path = "output/lightning_logs/version_2/checkpoints/epoch=9-step=7800.ckpt"
model_manager.load_lora(lora_weight_path, lora_alpha=1.0)

# 初始化 pipeline，多 GPU 环境下启用 usp 模式
pipe = WanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device=f"cuda:{rank}",
    use_usp=True if world_size > 1 else False
)
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# 指定存放原始 image、txt 和 ground-truth video 的文件夹路径
input_folder = "/project/osprey/scratch/x.zhexiao/video_gen/DiffSynth-Studio/data/example_physionv3_short"

# 指定生成的视频和 HTML 文件的输出文件夹路径（仅在 rank 0 负责创建）
output_folder = "/project/osprey/scratch/x.zhexiao/video_gen/DiffSynth-Studio/output/physionv3_short_4.5_7800"
if rank == 0:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
# 等待所有进程完成文件夹创建
dist.barrier()

# 获取所有 jpg 图片路径，并排序
all_image_paths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))

# 每个进程仅处理下标模 world_size 相等于自身 rank 的图片
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
    
    # 进行 Image-to-Video 推理
    video = pipe(
        prompt=full_prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=image,
        num_inference_steps=50,
        seed=0,
        tiled=True
    )
    
    # 保存生成的视频到输出文件夹
    dest = os.path.join(output_folder, base_name + "_generated.mp4")
    export_to_video(video, dest, fps=24)
    
    # 根据原始 image 路径生成 ground-truth 视频路径（保留在 input_folder 中）
    gt_video_path = image_path.rsplit(".", 1)[0] + ".mp4"
    
    # 将生成的视频和 ground-truth 视频编码为 base64 data URL
    gen_video_data_url = encode_video_to_data_url(dest)
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

# 收集所有进程的 HTML 条目到 rank 0
# all_html_entries = [None] * world_size
if rank == 0:
    all_html_entries = [None] * world_size
else:
    all_html_entries = None
dist.gather_object(local_html_entries, all_html_entries, dst=0)

if rank == 0:
    # 将所有子列表合并成一个列表
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

# 等待所有进程结束后清理分布式进程组
dist.barrier()
dist.destroy_process_group()

import os
import glob
import base64
import torch
from PIL import Image
from diffsynth import ModelManager, WanVideoPipeline, save_video
from modelscope import snapshot_download, dataset_snapshot_download

# 如果没有现成的 export_to_video 和 encode_video_to_data_url，可如下定义：
def export_to_video(video, dest, fps=24):
    # 这里使用 save_video 保存视频，可根据需要调整参数
    save_video(video, dest, fps=fps, quality=5)

def encode_video_to_data_url(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
    encoded = base64.b64encode(video_data).decode('utf-8')
    return f"data:video/mp4;base64,{encoded}"

# 模型下载（如已下载可注释掉）
# snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="models/Wan-AI/Wan2.1-I2V-14B-480P")

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
lora_weight_path = "output/lightning_logs/version_2/checkpoints/epoch=8-step=5400.ckpt"
model_manager.load_lora(lora_weight_path, lora_alpha=1.0)

# 初始化 pipeline 并配置显存管理
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)


# 指定存放原始 image、txt 和 ground-truth video 的文件夹路径
input_folder = "/project/osprey/scratch/x.zhexiao/video_gen/DiffSynth-Studio/data/example_physionv3_short"  # 请修改为实际路径

# 指定生成的视频和 HTML 文件的输出文件夹路径
output_folder = "/project/osprey/scratch/x.zhexiao/video_gen/DiffSynth-Studio/output/physionv3_short_4.5_5400"  # 请修改为实际输出路径
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 用于存储每个条目的 HTML 内容
html_entries = []

# 遍历文件夹中所有 jpg 图片
for image_path in glob.glob(os.path.join(input_folder, "*.jpg")):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(input_folder, base_name + ".txt")
    
    # 检查是否存在同名的 txt 文件
    if not os.path.exists(txt_path):
        print(f"Warning: 未找到 {txt_path}，跳过 {image_path}")
        continue

    # 读取 prompt 文本内容
    with open(txt_path, "r", encoding="utf-8") as f:
        full_prompt = f.read().strip()

    # 打开图片
    image = Image.open(image_path)
    
    # 进行 Image-to-Video 推理
    video = pipe(
        prompt=full_prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=image,
        num_inference_steps=50,
        seed=0,      # 可根据需要修改或随机化 seed
        tiled=True
    )
    
    # 保存生成的视频到输出文件夹（使用 export_to_video）
    dest = os.path.join(output_folder, base_name + "_generated.mp4")
    export_to_video(video, dest, fps=24)
    
    # 根据原始 image 路径生成 ground-truth 视频路径（扩展名改为 .mp4，保留在 input_folder 中）
    gt_video_path = image_path.rsplit(".", 1)[0] + ".mp4"
    
    # 将生成的视频和 ground-truth 视频编码为 base64 data URL
    gen_video_data_url = encode_video_to_data_url(dest)
    if os.path.exists(gt_video_path):
        gt_video_data_url = encode_video_to_data_url(gt_video_path)
    else:
        gt_video_data_url = ""
    
    # 构造 HTML 条目，每个条目显示 caption、生成视频和 ground-truth 视频
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
    html_entries.append(entry_html)
    print(f"已处理 {image_path}")

# 构造完整的 HTML 页面
html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Video Generation Results</title>
</head>
<body>
    {"".join(html_entries)}
</body>
</html>
'''

# 保存 HTML 文件到输出文件夹下
html_path = os.path.join(output_folder, "results.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"HTML 结果已保存至 {html_path}")

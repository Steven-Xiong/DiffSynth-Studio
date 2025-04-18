import json
import csv
import os

# 在此处直接指定 JSON 文件的路径和生成的 CSV 文件路径
json_file = '/project/osprey/scratch/x.zhexiao/video_gen/DiffSynth-Studio/data/physion/captions_v3_78B_short.json'
csv_file = '/project/osprey/scratch/x.zhexiao/video_gen/DiffSynth-Studio/data/physion1/metadata.csv'



# 读取 JSON 文件内容
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = []
for item in data:
    # 直接使用 JSON 中的 video_path 作为 file_name
    file_name = item["video_path"]
    # 对 caption 字段可以选择去掉可能多余的双引号，也可以直接使用
    text = item["caption"].strip('"')
    rows.append({"file_name": file_name, "text": text})

# 写入 CSV 文件
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['file_name', 'text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV 文件已生成：{csv_file}")



# # 读取 JSON 文件内容
# with open(json_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# rows = []
# # 遍历 JSON 中的每个记录，使用 1 开始的序号构造新的文件名
# for i, item in enumerate(data, start=1):
#     # 从 video_path 中提取原始文件名
#     original_filename = os.path.basename(item["video_path"])
#     # 获取文件扩展名
#     ext = os.path.splitext(original_filename)[1].lower()
    
#     # 根据扩展名判断文件类型，并确定前缀和新扩展名
#     if ext == '.mp4':
#         prefix = 'video'
#         new_ext = '.mp4'
#     else:
#         prefix = 'image'
#         new_ext = '.jpg'
    
#     # 生成新的文件名，序号用五位数字填充
#     new_filename = f"{prefix}_{i:05d}{new_ext}"
    
#     # 处理 caption 字段，去除可能多余的双引号
#     caption = item["caption"].strip('"')
    
#     rows.append({
#         "file_name": new_filename,
#         "text": caption
#     })

# # 写入 CSV 文件
# with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
#     fieldnames = ['file_name', 'text']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(rows)

# print(f"转换完成，CSV 文件已生成：{csv_file}")


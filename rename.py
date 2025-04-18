from pathlib import Path

folder = Path('/u/zhexiao/video_gen/DiffSynth-Studio/output/physionv3_short_4.5_6000')        # ← 换成你的目录

for f in folder.iterdir():
    if f.is_file() and f.stem.endswith('_'):
        # import pdb; pdb.set_trace()
        f.rename(f.with_name(f.stem[:-1] + f.suffix))  # 去掉 “_generated”

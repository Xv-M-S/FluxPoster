import os
import shutil

# 设置源目录和目标目录
source_dir = "/home/sxm/data02Space/DataSets/outputs"
target_dir = "/home/sxm/data02Space/DataSets/outputs_subset"

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 获取源目录下的所有子文件夹（排除文件）
folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

# 按文件夹名排序（自然排序，例如 "folder1", "folder2", ..., "folder100"）
folders.sort()

# 取前10000个文件夹
selected_folders = folders[:10000]

# 开始复制
for folder in selected_folders:
    source_path = os.path.join(source_dir, folder)
    target_path = os.path.join(target_dir, folder)

    print(f"Copying: {source_path} -> {target_path}")
    try:
        shutil.copytree(source_path, target_path)
    except Exception as e:
        print(f"Error copying {folder}: {e}")

print("✅ 复制完成。")
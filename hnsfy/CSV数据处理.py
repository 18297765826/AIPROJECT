"""
=========================================
功能说明：
-----------------------------------------
根据 CSV 中的映射关系：
    scan_file  →  fd_name

将源目录中的图片文件进行分类复制：
    按 fd_name 分类放入不同文件夹

特点：
- 自动清洗非法文件夹名称
- 忽略未在 CSV 中出现的文件
- 支持多层目录遍历
- 统计处理结果
=========================================
"""

import os
import shutil
import pandas as pd


# =========================
# 1. 路径配置
# =========================
# CSV 文件路径（包含映射关系）
csv_path = r'F:\分段数据\分段数据.csv'

# 原始图片根目录（按子文件夹存放）
src_root = r'F:\已检20001-20978已完成'

# 输出目录（分类结果）
dst_root = r'F:\分类结果0-20_按名称'

# 如果目标目录不存在则创建
os.makedirs(dst_root, exist_ok=True)


# =========================
# 2. 读取 CSV 数据
# =========================
# 使用 gbk 编码读取（常见于中文 Windows 环境）
df = pd.read_csv(csv_path, encoding='gbk')

# 删除关键字段为空的数据（避免后续报错）
df = df.dropna(subset=['fd_name', 'scan_file'])

# 去除字符串前后空格（非常关键，避免匹配失败）
df['fd_name'] = df['fd_name'].astype(str).str.strip()
df['scan_file'] = df['scan_file'].astype(str).str.strip()


# =========================
# 3. 清洗非法文件夹名称
# =========================
# Windows 文件夹不允许以下字符：
# \ / : * ? " < > |
# 否则创建目录会报错
def clean_name(name):
    return name.replace('/', '_').replace('\\', '_').replace(':', '_') \
               .replace('*', '_').replace('?', '_').replace('"', '_') \
               .replace('<', '_').replace('>', '_').replace('|', '_')

# 对所有分类名称进行清洗
df['fd_name'] = df['fd_name'].apply(clean_name)


# =========================
# 4. 构建映射关系
# =========================
# key：文件名（小写）
# value：分类名称 fd_name
# 使用小写是为了避免 Windows 大小写不一致问题
file_to_name = dict(zip(df['scan_file'].str.lower(), df['fd_name']))

print(f"映射数量: {len(file_to_name)}")


# =========================
# 5. 遍历源目录并复制文件
# =========================
total = 0      # 总扫描文件数
copied = 0     # 成功复制数
skipped = 0    # 被跳过数（无映射）

# 遍历第一层子目录
for folder in os.listdir(src_root):
    folder_path = os.path.join(src_root, folder)

    # 跳过非文件夹
    if not os.path.isdir(folder_path):
        continue

    # 遍历子目录中的文件
    for file in os.listdir(folder_path):
        total += 1

        # 转为小写用于匹配
        file_lower = file.lower()

        # 只处理图片文件
        if not file_lower.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 如果 CSV 中没有该文件 → 跳过
        if file_lower not in file_to_name:
            skipped += 1
            continue

        # 获取目标分类名称
        fd_name = file_to_name[file_lower]

        # 构建目标目录路径
        dst_dir = os.path.join(dst_root, fd_name)

        # 创建分类文件夹（若不存在）
        os.makedirs(dst_dir, exist_ok=True)

        # 源文件路径
        src_path = os.path.join(folder_path, file)

        # 目标文件路径
        dst_path = os.path.join(dst_dir, file)

        try:
            # 复制文件（保留元数据，如时间）
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception as e:
            print(f"复制失败: {file}, 错误: {e}")


# =========================
# 6. 输出统计结果
# =========================
print(f"总文件数: {total}")
print(f"成功复制: {copied}")
print(f"跳过(无映射): {skipped}")
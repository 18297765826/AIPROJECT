"""
=========================================
功能说明（指定类别 + 限量复制版本）
-----------------------------------------
从 CSV 映射中筛选指定类别的文件，并进行分类复制：

核心功能：
1️⃣ 只复制 TARGET_CLASSES 中定义的类别
2️⃣ 每个类别最多复制 MAX_PER_CLASS 张图片
3️⃣ 自动清洗非法文件夹名称
4️⃣ 输出分类统计结果

适用场景：
✔ 构建训练集（类别均衡）
✔ 抽样数据分析
✔ OCR/分类模型数据准备
=========================================
"""

import os
import shutil
import pandas as pd
from collections import defaultdict


# =========================
# 1. 配置
# =========================

# CSV路径（包含 scan_file → fd_name 映射）
csv_path = r'F:\分段数据\分段数据.csv'

# 原始图片目录（多子文件夹）
src_root = r'F:\已检20001-20978'

# 输出目录（分类结果）
dst_root = r'F:\分类结果_筛选版'

# 👉 指定要提取的类别（只处理这些类别）
TARGET_CLASSES = {
    '病危（重）通知书',
    '死亡病历讨论记录',
    '特殊检查（特殊治疗记录）',
    '新生儿记录',
    '住院通知单',
    '死亡记录',
    # '手术记录',  # 可扩展
}

# 👉 每个类别最多复制多少张
# None 表示不限制
MAX_PER_CLASS = 5000

# 创建输出目录
os.makedirs(dst_root, exist_ok=True)


# =========================
# 2. 读取 CSV 数据
# =========================

df = pd.read_csv(csv_path, encoding='gbk')

# 删除关键字段为空的数据
df = df.dropna(subset=['fd_name', 'scan_file'])

# 清洗字符串（去空格）
df['fd_name'] = df['fd_name'].astype(str).str.strip()
df['scan_file'] = df['scan_file'].astype(str).str.strip()


# =========================
# 3. 清理非法文件夹名称
# =========================
# Windows 文件夹不能包含特殊字符
def clean_name(name):
    return name.replace('/', '_').replace('\\', '_').replace(':', '_') \
               .replace('*', '_').replace('?', '_').replace('"', '_') \
               .replace('<', '_').replace('>', '_').replace('|', '_')

df['fd_name'] = df['fd_name'].apply(clean_name)


# =========================
# 4. 类别筛选（核心）
# =========================
# 👉 只保留目标类别的数据
df = df[df['fd_name'].isin(TARGET_CLASSES)]

# 查看筛选后的类别分布（非常重要）
print("筛选后类别统计：")
print(df['fd_name'].value_counts())


# =========================
# 5. 构建文件映射
# =========================
# key：文件名（小写）
# value：类别名称
file_to_name = dict(zip(df['scan_file'].str.lower(), df['fd_name']))


# =========================
# 6. 类别计数器（用于限量）
# =========================
# 默认值为0
class_counter = defaultdict(int)


# =========================
# 7. 遍历源目录并复制
# =========================
total = 0     # 扫描文件数
copied = 0    # 成功复制数
skipped = 0   # 无映射跳过
limited = 0   # 因数量限制跳过

# 遍历一级子目录
for folder in os.listdir(src_root):
    folder_path = os.path.join(src_root, folder)

    # 跳过非文件夹
    if not os.path.isdir(folder_path):
        continue

    # 遍历子目录文件
    for file in os.listdir(folder_path):
        total += 1

        file_lower = file.lower()

        # 只处理图片文件
        if not file_lower.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 不在映射表 → 跳过
        if file_lower not in file_to_name:
            skipped += 1
            continue

        # 获取类别
        fd_name = file_to_name[file_lower]

        # =========================
        # 👉 核心控制：类别数量限制
        # =========================
        if MAX_PER_CLASS is not None and class_counter[fd_name] >= MAX_PER_CLASS:
            limited += 1
            continue

        # 创建目标分类目录
        dst_dir = os.path.join(dst_root, fd_name)
        os.makedirs(dst_dir, exist_ok=True)

        # 构建路径
        src_path = os.path.join(folder_path, file)
        dst_path = os.path.join(dst_dir, file)

        try:
            # 复制文件（保留元数据）
            shutil.copy2(src_path, dst_path)

            copied += 1
            class_counter[fd_name] += 1  # 更新该类计数

        except Exception as e:
            print(f"复制失败: {file}, 错误: {e}")


# =========================
# 8. 结果统计
# =========================
print("\n===== 统计结果 =====")
print(f"总扫描文件: {total}")
print(f"成功复制: {copied}")
print(f"跳过(无映射): {skipped}")
print(f"因数量限制跳过: {limited}")

print("\n各类别数量：")
for k, v in class_counter.items():
    print(f"{k}: {v}")
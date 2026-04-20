import os
import random
import shutil
from pathlib import Path


def split_yolo_cls_dataset(src_dir, dst_dir, train_ratio=0.8, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    功能：
        将“按类别文件夹组织”的分类数据集，
        转换为 YOLO 分类格式，并划分 train / val

    原始结构（src_dir）:
        src_dir/
            ├── class1/
            │     ├── img1.jpg
            │     ├── img2.jpg
            ├── class2/
                  ├── img3.jpg

    输出结构（dst_dir）:
        dst_dir/
            ├── train/
            │     ├── class1/
            │     ├── class2/
            ├── val/
                  ├── class1/
                  ├── class2/

    参数:
        src_dir (str or Path): 原始数据目录
        dst_dir (str or Path): 输出目录
        train_ratio (float): 训练集比例 (默认 0.8)
        exts (tuple): 支持的图片格式
    """

    # 转换为 Path 对象（更现代、更安全）
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # 定义训练集和验证集目录
    train_dir = dst_dir / "train"
    val_dir = dst_dir / "val"

    # 遍历每一个类别文件夹
    for class_dir in src_dir.iterdir():

        # 跳过非目录（避免误读文件）
        if not class_dir.is_dir():
            continue

        # 类别名称（文件夹名即类别名）
        class_name = class_dir.name

        # 在目标路径中创建类别目录（train / val）
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)

        # 收集当前类别下的所有图片文件
        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in exts  # 过滤合法图片格式
        ]

        # 如果该类别没有图片，跳过
        if len(images) == 0:
            continue

        # 随机打乱数据（避免顺序偏差）
        random.shuffle(images)

        # 计算划分位置
        split_idx = int(len(images) * train_ratio)

        # 划分训练集和验证集
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # =========================
        # 复制训练集图片
        # =========================
        for img in train_imgs:
            shutil.copy(
                img,  # 源文件路径
                train_dir / class_name / img.name  # 目标路径
            )

        # =========================
        # 复制验证集图片
        # =========================
        for img in val_imgs:
            shutil.copy(
                img,
                val_dir / class_name / img.name
            )

        # 打印当前类别统计信息
        print(f"{class_name}: train={len(train_imgs)}, val={len(val_imgs)}")

    print("✅ 数据集划分完成！")


# =========================
# 主程序入口
# =========================
if __name__ == "__main__":

    split_yolo_cls_dataset(
        src_dir=r"F:\hnsfy20",        # 原始数据路径
        dst_dir=r"F:\hnsfy20_yolo",   # 输出路径
        train_ratio=0.8               # 训练集比例
    )
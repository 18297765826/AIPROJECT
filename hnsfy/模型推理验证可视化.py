import os
import shutil
from ultralytics import YOLO
import torch

# ==============================
# 配置
# ==============================
# 模型路径（YOLO分类模型）
# MODEL_PATH = r"D:\Paddle\runs\classify\train24\weights\last.pt"
MODEL_PATH = r"F:\last.pt"

# 待推理图片目录（支持递归）
SRC_DIR = r"F:\420test"

# 推理结果输出目录
DST_DIR = r"F:\420test_results"

# 每次推理的批量大小（影响速度/显存）
BATCH_SIZE = 32


# ==============================
# 类别映射（模型类别 → 中文名称）
# ==============================
# 注意：
# model.names 里通常是数字字符串（如 '1','2'）
# 这里做一次“业务语义映射”
class_map = {
    '30': '其他',
    '1': '住院病案首页',
    '2': '入院记录',
    '8': '病程记录',
    '9': '术前讨论记录',
    '33': '术前风险评估',
    '10': '手术同意书',
    '11': '麻醉同意书',
    '12': '麻醉术前访视记录',
    '13': '手术安全核查记录',
    '14': '手术清点记录',
    '15': '麻醉记录',
    '5': '手术记录',
    '4': '分娩记录',
    '16': '麻醉术后访视记录',
    '17': '术后病程记录',
    '3': '出院记录',
    '20': '输血治疗同意书',
    '31': '其他同意书',
    '22': '会诊记录',
    '24': '辅助检查报告单',
    '25': '医学影像检查资料',
    '28': '病重（病危）患者护理记录',
    '26': '体温单',
    '6': '医嘱单',
    '27': '评估评分单',
    '23': '病危（重）通知书',
    '7': '病理资料',
    '21': '特殊检查（特殊治疗记录）',
    '19': '死亡病历讨论记录',
    '29': '新生儿记录',
    '32': '住院通知单',
    '18': '死亡记录'
}


# ==============================
# 工具函数
# ==============================

def get_all_images(root_dir):
    """
    递归获取目录下所有图片路径

    参数:
        root_dir: 根目录

    返回:
        List[str]: 图片路径列表
    """
    img_paths = []

    # os.walk 会递归遍历所有子目录
    for root, _, files in os.walk(root_dir):
        for f in files:
            # 过滤图片格式
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_paths.append(os.path.join(root, f))

    return img_paths


def safe_path(path):
    """
    防止文件名冲突，如果已存在则自动重命名

    例如:
        a.jpg -> a_1.jpg -> a_2.jpg ...

    参数:
        path: 原始路径

    返回:
        不冲突的新路径
    """
    base, ext = os.path.splitext(path)
    i = 1
    new_path = path

    # 如果路径已存在，就不断加后缀
    while os.path.exists(new_path):
        new_path = f"{base}_{i}{ext}"
        i += 1

    return new_path


# ==============================
# 主函数
# ==============================
def main():

    # ==========================
    # 1. 加载模型
    # ==========================
    print("🚀 加载模型...")
    model = YOLO(MODEL_PATH)

    # 打印类别顺序（非常关键，用于调试映射）
    print("📌 模型类别顺序:", model.names)

    # 设置CPU线程数（防止过度占用资源）
    torch.set_num_threads(4)

    # ==========================
    # 2. 收集图片
    # ==========================
    img_paths = get_all_images(SRC_DIR)
    print(f"📂 共找到 {len(img_paths)} 张图片")

    if not img_paths:
        print("❌ 没有图片")
        return

    print("🧠 开始推理...")

    # ==========================
    # 3. 批量推理（提升性能）
    # ==========================
    for i in range(0, len(img_paths), BATCH_SIZE):

        # 当前批次
        batch = img_paths[i:i+BATCH_SIZE]

        print(f"🚀 进度 {i}/{len(img_paths)}")

        try:
            # YOLO 批量预测
            results = model.predict(
                batch,
                verbose=False,   # 关闭详细日志
                workers=0        # 防止多进程冲突（Windows常见坑）
            )
        except Exception as e:
            print(f"❌ 批次失败: {e}")
            continue

        # ==========================
        # 4. 处理每一张图片结果
        # ==========================
        for img_path, result in zip(batch, results):
            try:
                # 原始文件名
                img_name = os.path.basename(img_path)

                # ======================
                # 4.1 获取预测类别
                # ======================
                cls_id = int(result.probs.top1)   # 最高概率类别ID
                cls_label = str(model.names[cls_id])  # 模型类别名（字符串）
                cls_name = class_map.get(cls_label, "未知")  # 映射中文

                # ======================
                # 4.2 构造新文件名
                # ======================
                name, ext = os.path.splitext(img_name)

                # 防止重复添加类别标签
                if any(v in name for v in class_map.values()):
                    new_name = f"{name}{ext}"
                else:
                    new_name = f"{name}_{cls_name}{ext}"

                # ======================
                # 4.3 保持原始目录结构
                # ======================
                rel_dir = os.path.relpath(os.path.dirname(img_path), SRC_DIR)
                dst_folder = os.path.join(DST_DIR, rel_dir)

                os.makedirs(dst_folder, exist_ok=True)

                # ======================
                # 4.4 复制文件
                # ======================
                dst_path = os.path.join(dst_folder, new_name)

                # 防止重名覆盖
                dst_path = safe_path(dst_path)

                shutil.copy(img_path, dst_path)

                print(f"✅ {img_name} → {new_name}")

            except Exception as e:
                print(f"❌ 单张失败: {img_path}, 错误: {e}")

    print("🎉 全部完成！")


# ==============================
# 程序入口
# ==============================
if __name__ == "__main__":
    main()
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import warnings

warnings.filterwarnings('ignore')

# ==========  ==========
IMAGES_DIR = Path("dir/images")
OUTPUT_CSV = "dir/result/shenzhen_seg_all.xlsx"
LOCAL_MODEL_PATH =  r"mask2former_model_cityscapes"
SAVE_VISUALIZATION = True  #

# Cityscapes
CLASS_ID_TO_NAME = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
    5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
}

# color
COLOR_MAP = {
    10: [70, 130, 180],  #
    8: [107, 142, 35],  #
    0: [128, 64, 128],  #
    2: [70, 70, 70],  #
}

# ==========  ==========
print("...")
processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH, use_fast=False)
model = Mask2FormerForUniversalSegmentation.from_pretrained(LOCAL_MODEL_PATH)
model.eval()
print("finish")


# ========== main ==========
def split_panorama(panorama_img):
    """four images"""
    width, height = panorama_img.size
    single_width = width // 4
    return [panorama_img.crop((i * single_width, 0, (i + 1) * single_width, height)) for i in range(4)]


def get_pixel_counts_and_map(image, model, processor):
    """statistic"""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    seg_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0].cpu().numpy()

    total_pixels = seg_map.size
    unique_classes, counts = np.unique(seg_map, return_counts=True)

    counts_dict = {'total_pixels': int(total_pixels)}
    for class_id, count in zip(unique_classes, counts):
        class_name = CLASS_ID_TO_NAME.get(class_id, f'class_{class_id}')
        counts_dict[class_name] = int(count)

    return counts_dict, seg_map


def save_visualization(original_img, seg_map, direction, filename):
    """"""
    # seg-images
    h, w = seg_map.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in COLOR_MAP.items():
        color_map[seg_map == class_id] = color

    #
    from matplotlib import pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(original_img)
    ax1.set_title(f' - {direction}')
    ax1.axis('off')

    ax2.imshow(color_map)
    ax2.set_title(f' - {direction}')
    ax2.axis('off')

    plt.tight_layout()
    output_path = f"visualization_{filename}_{direction}.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  : {output_path}")


def process_single_panorama(panorama_path, model, processor, save_viz=False, is_first=False):
    """process"""
    panorama_img = Image.open(panorama_path).convert('RGB')
    sub_imgs = split_panorama(panorama_img)

    directions = ['0', '90', '180', '270']
    direction_stats = []
    aggregated = {'total_pixels': 0}

    #
    for class_name in CLASS_ID_TO_NAME.values():
        aggregated[class_name] = 0

    for i, (sub_img, direction) in enumerate(zip(sub_imgs, directions)):
        counts, seg_map = get_pixel_counts_and_map(sub_img, model, processor)
        counts['direction'] = direction

        #
        aggregated['total_pixels'] += counts['total_pixels']
        for class_name in CLASS_ID_TO_NAME.values():
            aggregated[class_name] += counts.get(class_name, 0)

        direction_stats.append(counts)

        # save
        if save_viz and is_first:
            save_visualization(sub_img, seg_map, direction, panorama_path.stem)

    return direction_stats, aggregated


def parse_filename(filename):
    """location"""
    try:
        parts = filename.stem.split('_')
        return float(parts[0]), float(parts[1])
    except:
        return None, None


# ========== ==========
print("start...")

#
image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png']:
    image_files.extend(list(IMAGES_DIR.glob(ext)))

if not image_files:
    print("no images")
    exit(1)

#
all_data = []
first_image_processed = False

for i, img_path in enumerate(image_files):
    print(f"start [{i + 1}/{len(image_files)}]: {img_path.name}")

    lon, lat = parse_filename(img_path)
    if lon is None:
        continue

    #
    direction_stats, aggregated = process_single_panorama(
        img_path, model, processor,
        save_viz=SAVE_VISUALIZATION,
        is_first=(not first_image_processed)
    )

    if not first_image_processed:
        first_image_processed = True

    #
    for stats in direction_stats:
        row = {
            'filename': img_path.name,
            'longitude': lon,
            'latitude': lat,
            'direction': stats['direction'],
            'total_pixels': stats['total_pixels']
        }
        #
        for class_name in CLASS_ID_TO_NAME.values():
            row[class_name] = stats.get(class_name, 0)
        all_data.append(row)

    #
    agg_row = {
        'filename': f"{img_path.stem}_all",
        'longitude': lon,
        'latitude': lat,
        'direction': 'all',
        'total_pixels': aggregated['total_pixels']
    }
    for class_name in CLASS_ID_TO_NAME.values():
        agg_row[class_name] = aggregated.get(class_name, 0)
    all_data.append(agg_row)

# ========== result ==========
if all_data:
    df = pd.DataFrame(all_data)

    # 重新排序列
    base_cols = ['filename', 'longitude', 'latitude', 'direction', 'total_pixels']
    class_cols = [CLASS_ID_TO_NAME[i] for i in sorted(CLASS_ID_TO_NAME.keys())]
    df = df[base_cols + class_cols]

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n！ {len(image_files)} images")
    print(f"save: {OUTPUT_CSV}")

    # 显示前几行数据
    print("\ndata:")
    print(df.head().to_string())
else:
    print("no data")
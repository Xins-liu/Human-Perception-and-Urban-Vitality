"""

"""
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import shutil
import json
import os
import sys
#
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
    os.environ.pop(key, None)  #

# 2.
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# 3.
#
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

print("。")
# ====  ====
# ==========  ==========
IMAGES_DIR = Path("dir/images")
ALL_SUBIMAGES_DIR = Path("dir/all_subimages")
OUTPUT_CSV = "dir/result/shenzhen_perception_scores.csv"
#OUTPUT_CSV = "dir/result/shenzhen_perception_scores.csv"
PERCEPTION_INDICATORS = ['more beautiful', 'more boring', 'more depressing',
                         'livelier', 'safer', 'wealthier']


def split_panoramas_to_subimages(panorama_files, output_dir):
    """


    Args:
        panorama_files:
        output_dir:

    Returns:
        DataFrame
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    records = []

    for pano_path in panorama_files:
        try:
            # .jpg
            lon_str, lat_str, _ = pano_path.stem.split('_', 2)
            lon, lat = float(lon_str), float(lat_str)
        except ValueError:
            print(f"  next）: {pano_path.name}")
            continue

        img = Image.open(pano_path).convert('RGB')
        width, height = img.size
        single_width = width // 4
        directions = ['0度', '90度', '180度', '270度']

        for i, dir_name in enumerate(directions):
            left = i * single_width
            sub_img = img.crop((left, 0, left + single_width, height))

            # .jpg
            sub_name = f"{pano_path.stem}_{dir_name}.jpg"
            sub_path = output_dir / sub_name
            sub_img.save(sub_path, quality=90)

            records.append({
                'subimage_name': sub_name,
                'panorama_name': pano_path.name,
                'longitude': lon,
                'latitude': lat,
                'direction': dir_name
            })

        img.close()
        print(f"  finish: {pano_path.name}")

    if records:
        mapping_df = pd.DataFrame(records)
        mapping_df.to_csv(output_dir / "00_subimage_mapping.csv", index=False)
        print(f"  images: {len(mapping_df)} ")
        return mapping_df
    else:
        print("  error no images")
        return pd.DataFrame()


def calculate_perception_scores(indicators, subimage_dir, mapping_df):
    """


    Args:
        indicators:
        subimage_dir:
        mapping_df:

    Returns:
        dict:  {Indicators: {name: score}}
    """
    from zensvi.cv import ClassifierPerception

    all_scores = {}

    for indicator in indicators:
        print(f"  caculate: {indicator}")
        temp_dir = subimage_dir / f"temp_{indicator}"
        temp_dir.mkdir(exist_ok=True)

        try:
            classifier = ClassifierPerception(perception_study=indicator)
            classifier.classify(
                str(subimage_dir),
                dir_summary_output=str(temp_dir),
                batch_size=4
            )

            result_file = find_result_file(temp_dir)
            if result_file:
                scores = extract_scores_from_file(result_file)
                all_scores[indicator] = scores
                print(f"    success: {len(scores)} images")
            else:
                print(f"    no result")
                all_scores[indicator] = {}

        except Exception as e:
            print(f"    error compute: {e}")
            all_scores[indicator] = {}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return all_scores


def find_result_file(directory):
    """"""
    for suffix in ['.csv', '.json', '.txt']:
        files = list(directory.glob(f"*{suffix}"))
        if files:
            return files[0]
    return None


def extract_scores_from_file(result_file):
    """

    """
    scores_map = {}

    try:
        if result_file.suffix == '.csv':
            df = pd.read_csv(result_file)

            #
            filename_col = 'filename_key'
            #
            score_col = [c for c in df.columns if c != filename_col][0] if len(df.columns) > 1 else None

            if not score_col:
                print(f"      {result_file} no score")
                return scores_map

            for _, row in df.iterrows():
                raw_name = str(row[filename_col])  #
                #
                filename = f"{raw_name}.jpg"
                try:
                    scores_map[filename] = float(row[score_col])
                except (ValueError, TypeError) as e:
                    continue

        elif result_file.suffix == '.json':
            #
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            #
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        scores_map[f"{key}.jpg"] = float(value)
        else:
            print(f"      error: {result_file.suffix}")

    except Exception as e:
        print(f"      error ({result_file.name}): {e}")

    print(f"       {len(scores_map)} score。")  #
    return scores_map

def aggregate_scores_to_panoramas(mapping_df, all_scores):
    """


    Args:
        mapping_df:
        all_scores:

    Returns:
        DataFrame:
    """
    records = []

    #
    for (pano_name, lon, lat), group in mapping_df.groupby(['panorama_name', 'longitude', 'latitude']):
        record = {
            'panorama_name': pano_name,
            'longitude': lon,
            'latitude': lat,
        }

        # avg
        for indicator in PERCEPTION_INDICATORS:
            direction_scores = []  # avglist
            indicator_scores_dict = all_scores.get(indicator, {})

            # four images
            for _, subimage_row in group.iterrows():
                sub_name = subimage_row['subimage_name']
                dir_name = subimage_row['direction']

                # 1.
                if sub_name in indicator_scores_dict:
                    score = indicator_scores_dict[sub_name]
                    direction_scores.append(score)
                    # 2.
                    record[f'{dir_name}_{indicator}'] = score
                else:
                    # None
                    record[f'{dir_name}_{indicator}'] = None

            #
            if direction_scores:
                avg_score = np.mean(direction_scores)  #
                record[f'avg_{indicator}'] = avg_score  #
            else:
                record[f'avg_{indicator}'] = None

        records.append(record)

    return pd.DataFrame(records)

def main():
    """main"""
    print("=" * 60)
    print("IA")
    print("=" * 60)

    # 1. 准备输入数据
    panorama_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    if not panorama_files:
        print(f"error: {IMAGES_DIR} no images")
        return

    print(f"images: {len(panorama_files)} s\n")

    # 2.
    print("1: four images")
    mapping_df = split_panoramas_to_subimages(panorama_files, ALL_SUBIMAGES_DIR)
    if mapping_df.empty:
        return

    # 3.
    print("\nt2: caculate")
    all_scores = calculate_perception_scores(PERCEPTION_INDICATORS, ALL_SUBIMAGES_DIR, mapping_df)

    # 4. 聚合结果
    print("\nt3: score")
    result_df = aggregate_scores_to_panoramas(mapping_df, all_scores)

    # 5. 保存和报告
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("✅ finish！")
    print(f"   output: {OUTPUT_CSV}")
    print(f"   images len: {len(result_df)}")
    print(f"   IAindicators: {len(PERCEPTION_INDICATORS)} 个")

    # 成功率统计
    print(f"\nsuccessfulrate:")
    for indicator in PERCEPTION_INDICATORS:
        if indicator in all_scores:
            success = sum(1 for name in mapping_df['subimage_name']
                          if name in all_scores[indicator])
            total = len(mapping_df)
            print(f"  {indicator}: {success}/{total} ({success / total * 100:.1f}%)")

    #
    print(f"\nresult:")
    print(result_df.head().to_string(index=False))

    # 清理选项
    if input("\ndelete four images？(y/N): ").lower() == 'y':
        shutil.rmtree(ALL_SUBIMAGES_DIR, ignore_errors=True)
        print("finish。")
    else:
        print(f"remain: {ALL_SUBIMAGES_DIR}")


if __name__ == "__main__":
    main()
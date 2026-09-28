import re, os
import json
import requests
import time, glob
import csv
import traceback
from PIL import Image
from io import BytesIO


# write csv
def write_csv(filepath, data, head=None):
    if head:
        data = [head] + data
    with open(filepath, 'w', encoding='UTF-8-sig', newline='') as f:
        writer = csv.writer(f)
        for i in data:
            writer.writerow(i)

# read csv
def read_csv(filepath):
    data = []
    if os.path.exists(filepath):
        # 尝试多种编码方式
        encodings = ['UTF-8-sig', 'utf-8', 'gbk']
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    lines = csv.reader(f)
                    for line in lines:
                        data.append(line)
                print(f"成功使用 {encoding} 编码读取文件")
                return data
            except UnicodeDecodeError:
                print(f"{encoding} 编码读取失败，尝试下一种编码")
                data = []
                continue
        print('无法读取文件，请检查文件编码：{}'.format(filepath))
        return []
    else:
        print('文件不存在：{}'.format(filepath))
        return []


def grab_img_baidu(_url, _headers=None):
    if _headers is None:
        headers = {
            "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="90", "Google Chrome";v="90"',
            "Referer": "https://map.baidu.com/",
            "sec-ch-ua-mobile": "?0",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        }
    else:
        headers = _headers
    try:
        response = requests.get(_url, headers=headers, timeout=30)
        if response.status_code == 200 and response.headers.get('Content-Type') == 'image/jpeg':
            return response.content
        else:
            print(f"图片下载失败，状态码：{response.status_code}，内容类型：{response.headers.get('Content-Type')}")
            return None
    except Exception as e:
        print(f"图片下载异常：{e}")
        return None


def openUrl(_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    try:
        response = requests.get(_url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            print(f"URL请求失败，状态码：{response.status_code}")
            return None
    except Exception as e:
        print(f"URL请求异常：{e}")
        return None


def getPanoId(_lng, _lat):
    url = "https://mapsv0.bdimg.com/?&qt=qsdata&x=%s&y=%s&l=17.031000000000002&action=0&mode=day&t=1530956939770" % (
        str(_lng), str(_lat))
    response_content = openUrl(url)
    if response_content is None:
        return None
    try:
        response = response_content.decode("utf8")
    except:
        print("响应解码失败")
        return None

    if response is None:
        return None
    reg = r'"id":"(.+?)",'
    pat = re.compile(reg)
    try:
        svid = re.findall(pat, response)[0]
        return svid
    except:
        print("无法从响应中提取街景ID")
        return None



def wgs2bd09mc(wgs_x, wgs_y):
    # BaiduAK
    baidu_ak = '..........'#The AK number is private and cannot be made public

    url = 'http://api.map.baidu.com/geoconv/v1/?coords={}&from=1&to=6&output=json&ak={}'.format(
        wgs_x + ',' + wgs_y, baidu_ak)

    res_content = openUrl(url)
    if res_content is None:
        return None, None

    try:
        res = res_content.decode()
        temp = json.loads(res)
        if temp['status'] == 0:
            bd09mc_x = temp['result'][0]['x']
            bd09mc_y = temp['result'][0]['y']
            return bd09mc_x, bd09mc_y
        else:
            print(f"坐标转换API返回错误: {temp.get('message', '未知错误')}")
            return None, None
    except Exception as e:
        print(f"坐标转换异常: {e}")
        return None, None


# four images
def merge_images_horizontally(images_data, output_path):
    """
    水平拼接四张图片形成全景图
    顺序：0°(北) → 90°(东) → 180°(南) → 270°(西)
    """
    try:
        # 将二进制数据转换为PIL Image对象
        images = [Image.open(BytesIO(img_data)) for img_data in images_data]

        # 获取单张图片尺寸
        width, height = images[0].size

        # 创建新画布 (水平排列，宽度为4倍单图宽度)
        total_width = 4 * width
        merged_image = Image.new('RGB', (total_width, height))

        # 水平排列图片
        # 顺序：0°(北) → 90°(东) → 180°(南) → 270°(西)
        x_offset = 0
        for img in images:
            merged_image.paste(img, (x_offset, 0))
            x_offset += width

        # 保存合并后的图片
        merged_image.save(output_path, 'PNG')
        print(f"成功创建水平拼接全景图: {output_path}")

        # 关闭所有图片
        for img in images:
            img.close()

        return True

    except Exception as e:
        print(f"图片水平拼接失败: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    root = r'.\dir'
    read_fn = r'point_coordinate_intersect.csv'
    #
    mis_fn=r'sz_missing_points.csv'

    error_fn = r'error_road_intersection_shenzhen.csv'
    dir = r'images'

    #
    os.makedirs(os.path.join(root, dir), exist_ok=True)

    #
    filenames_exist = glob.glob1(os.path.join(root, dir), "*_panorama.png")

    #
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    data = read_csv(os.path.join(root, read_fn))
    #data = read_csv(os.path.join(root, mis_fn))
    if not data:
        print("无法读取CSV文件，程序退出")
        exit()

    header = data[0]
    data = data[1:]

    # 记录爬取失败的图片
    error_img = []
    # 记录没有svid的位置
    svid_none = []

    headings = ['0', '90', '180', '270']  # directions, 0 is north
    pitchs = '0'

    count = 1
    processed_count = 0


#
    #for i in range(50338, len(data)):
    for i in range(0,len(data)):
        print('Processing No. {} point...'.format(i + 1))
        #
        #
        #if len(data[i]) >= 13:  #
        if len(data[i]) >= 3:  #
            wgs_x, wgs_y = data[i][0], data[i][1]
            #mis的文件格式
            #wgs_x, wgs_y = data[i][12], data[i][13]

        else:
            print(f"数据行 {i} 列数不足，跳过")
            continue

        try:
            bd09mc_x, bd09mc_y = wgs2bd09mc(wgs_x, wgs_y)
            if bd09mc_x is None or bd09mc_y is None:
                print("坐标转换失败")
                error_img.append(data[i] + ['coord_convert_failed'])
                continue
        except Exception as e:
            print(f"坐标转换异常: {e}")
            error_img.append(data[i] + ['coord_convert_exception'])
            continue

        # 检查全景图片是否已存在
        panorama_filename = f"{wgs_x}_{wgs_y}_panorama.png"
        if panorama_filename in filenames_exist:
            print(f"全景图片已存在，跳过: {panorama_filename}")
            processed_count += 1
            continue

        svid = getPanoId(bd09mc_x, bd09mc_y)
        print(f"获取到街景ID: {svid}")

        if not svid:
            print(f"无法获取街景ID，跳过该点: {wgs_x}, {wgs_y}")
            svid_none.append([wgs_x, wgs_y])
            error_img.append(data[i] + ['no_svid'])
            continue


        images_data = []
        download_success = True

        for heading in headings:
            url = f'https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=100&panoid={svid}&heading={heading}&pitch=0&width=480&height=320'
            img_data = grab_img_baidu(url)

            if img_data is None:
                print(f"下载失败: heading={heading}")
                data[i].append(heading)
                error_img.append(data[i] + [f'download_failed_heading_{heading}'])
                download_success = False
                break
            else:
                images_data.append(img_data)
                print(f"下载成功: heading={heading}")


        if download_success and len(images_data) == 4:
            panorama_path = os.path.join(root, dir, panorama_filename)
            if merge_images_horizontally(images_data, panorama_path):
                print(f"成功创建全景图: {wgs_x}, {wgs_y}")
                processed_count += 1
            else:
                print(f"图片水平拼接失败: {wgs_x}, {wgs_y}")
                error_img.append(data[i] + ['panorama_merge_failed'])
        else:
            print(f"有图片下载失败，跳过拼接: {wgs_x}, {wgs_y}")


        print(f"等待6秒... ({processed_count}/{len(data[36066:])} 已完成)")#修改实现断点连续
        #print(f"等待6秒... ({processed_count}/{len(data[50337:])} 已完成)")
        time.sleep(6)
        count += 1


    if len(error_img) > 0:
        write_csv(os.path.join(root, error_fn), error_img, header + ['error_type'])

    # 保存没有svid的记录
    if len(svid_none) > 0:
        write_csv(os.path.join(root, 'no_svid_points.csv'), svid_none, ['wgs_x', 'wgs_y'])

    print(f"处理完成！成功处理 {processed_count} 个点，失败 {len(error_img)} 个点")
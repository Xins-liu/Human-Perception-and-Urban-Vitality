
import math
import csv
import json
import re
import requests
import time
import pandas as pd

#初始准备
SAVE_DATABASE = False #
if SAVE_DATABASE:
    from sqlalchemy import create_engine
    ENGINE = create_engine('postgresql://postgres:123@localhost:5432/weibo_crawler')
    TABLE_NAME1 = 'weibo_poi_shanghai'  #
    TABLE_NAME2 = 'weibo_poi_shanghai_detail'  #
else :
    CSV_PATH1 = 'data/weibo_poi_shenzhen.csv'
    CSV_PATH2 = 'data/weibo_poi_shenzhen_detail.csv'



DESCRIBE_URL = 'https://weibo.com/p/100101'  #

with open('config/describe_poi.txt', 'r', encoding='utf-8') as f:
    DESCRIBE_HEADERS = json.loads(f.read())  # poi  headers
    #print(DESCRIBE_HEADERS)
#with open('config/search_poi.txt', 'r', encoding='utf-8') as f:
    #SEARCH_HEADERS = json.loads(f.read())  #  headers
    #print(SEARCH_HEADERS)



def get_poi_detail(poiid: str) -> str:
    """poidetail"""

    # poidetail
    target_url = DESCRIBE_URL + str(poiid)
    pattern = re.compile(r'<p class=\\"p_txt\\">(.*?)<\\/p>', re.DOTALL)
    for attempt in range(1, 6):
        try:
            response = requests.get(target_url, headers=DESCRIBE_HEADERS)
            matches = pattern.findall(response.text)
            if matches:
                try:
                    category = matches[0].split(': ')[1]
                except Exception:
                    category = 'no'
                try:
                    address = matches[1].split(': ')[1]
                except Exception:
                    address = 'no'
                try:
                    dscribe = matches[2].split(': ')[1]
                except Exception:
                    dscribe = 'no'
                if SAVE_DATABASE:
                    #
                    data = {
                        'poiid': poiid,
                        'category': category,
                        'address': address,
                        'describe': dscribe,
                    }
                    row = pd.DataFrame(data, index=[0])
                    row.to_sql(TABLE_NAME2, con=ENGINE, if_exists='append', index=False)
                    break  # skip
                else:
                    # describe
                    with open(CSV_PATH2, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=['poiid', 'category', 'address', 'describe'])
                        if csvfile.tell() == 0:
                            writer.writeheader()
                        writer.writerow({'poiid': poiid, 'category': category, 'address': address, 'describe': dscribe})
                        print("successful")
                    break  # skip
            print(f"No text found for POI ID {poiid} on attempt {attempt}.")
        except Exception as e:
            print(f"Attempt {attempt} failed for POI ID {poiid}: {e}")
            time.sleep(80 * attempt)




poi1 = pd.read_csv('data/weibo_poi_shenzhen_cleaned_trans_wgs84.csv',encoding="gbk")
for poiid in poi1['poiid']:
    get_poi_detail(poiid)  #


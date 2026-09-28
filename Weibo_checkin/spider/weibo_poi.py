import csv
import json
import re
import requests
import time
import pandas as pd
import random

SAVE_DATABASE = False #
if SAVE_DATABASE:
    from sqlalchemy import create_engine
    ENGINE = create_engine('postgresql://postgres:123@localhost:5432/weibo_crawler')
    TABLE_NAME1 = 'weibo_poi_shanghai'  # POIinform
    TABLE_NAME2 = 'weibo_poi_shanghai_detail'  # POIdetail
else :
    CSV_PATH1 = 'data/weibo_poi_shenzhen.csv'
    CSV_PATH2 = 'data/weibo_poi_shenzhen_detail.csv'


SEARCH_URL = 'https://place.weibo.com/wandermap/search2'  #
POI_LIST_URL = "https://m.weibo.cn/api/container/getIndex"  #
DESCRIBE_URL = 'https://weibo.com/p/100101'  #  POIID
POI_CATEGORY = [26,27,28]  # weibopoi

with open('config/describe_poi.txt', 'r', encoding='utf-8') as f:
    DESCRIBE_HEADERS = json.loads(f.read())  # poi  headers
    #print(DESCRIBE_HEADERS)
#with open('config/search_poi.txt', 'r', encoding='utf-8') as f:
    #SEARCH_HEADERS = json.loads(f.read())  #  headers
    #print(SEARCH_HEADERS)
with open('config/params.txt', 'r', encoding='utf-8') as f:
    PARAMS = json.loads(f.read())  # poi
    #print(PARAMS)

# User-Agent
user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]

def get_poi(category_id: int, page: int) -> None:
    """get poi"""
    # poi name
    containerid = f'23044100{category_id}__8008644030000000000'
    PARAMS["containerid"] = containerid
    PARAMS['page'] = page
    try:
        headers = {
            'User-Agent': random.choice(user_agents),
            'Referer': 'https://m.weibo.cn',
            "Cookie": " "

        }
        response = requests.get(url=POI_LIST_URL, params=PARAMS,headers=headers)
        data = response.json()
        #print(data)
        if data['ok'] == 1:
            groups = data["data"]["cards"][-1]["card_group"]
            poi_names = [group['title_sub'] for group in groups]  # poi name
    except Exception:
        pass
    # search poi     ）））））））））））））））））））））））））））））））））））））））））））））））））））0000
    for keyword in poi_names:
        try:
            # mobile
            search_headers = {
                'User-Agent': "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1 Edg/142.0.0.0",
                "Cookie": '',
                'Referer': 'https://place.weibo.com/wandermap/search',
            }
            response = requests.get(SEARCH_URL, params={'keyword': keyword}, headers=search_headers)
            data = response.json()
            if data.get("pois"):
                print(f'page({page}):keyword\'{keyword}\'searched{len(data["pois"])}data')
                pois = data["pois"]
                if SAVE_DATABASE:
                    data_to_database(pois)  # save
                else:
                    data_to_csv(pois)  # save
        except Exception as e:
            print(f'Keyword {keyword} search failed: {e}')




def data_to_database(datas):
    """s"""
    for item in datas:
        try:
            row = pd.DataFrame({
                'poiid': [item['poiid']],
                'title': [item['title']],
                'lon': [item['lon']],
                'lat': [item['lat']],
                'address': [item['address']],
            })
            row.to_sql(TABLE_NAME1, con=ENGINE, if_exists='append', index=False)
        except Exception as e:
            print("Write to Database ERROR", e)

def data_to_csv(datas):
    """save csv"""
    fieldnames = ['poiid', 'title', 'lon', 'lat', 'address', 'describe']
    with open(CSV_PATH1, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # head
        if csvfile.tell() == 0:
            writer.writeheader()
        for item in datas:
            row = {
                'poiid': item['poiid'],
                'title': item['title'],
                'lon': item['lon'],
                'lat': item['lat'],
                'address': item['address'],
            }
            writer.writerow(row)



def save_params(page=1, containerid='2304410024__8008644030000000000'):
    """parama"""
    PARAMS["page"] = page
    PARAMS["containerid"] = containerid
    with open('config/params.txt', 'w', encoding='utf-8') as file:
        json.dump(PARAMS, file)



#主程序
for Category in POI_CATEGORY:
    print(f'poi category {Category} data...')
    page = 1  #
    while True:
        try:
            get_poi(Category, page)
            page += 1
            save_params(page=page, containerid=f'23044100{Category}__8008644030000000000')
            time.sleep(5)  # sleep
        except Exception as e:
            print(f"Error occurred: {e}")
            break

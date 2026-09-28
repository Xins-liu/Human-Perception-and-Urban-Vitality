import time
import requests
import pandas as pd
import random
import json
import os
import logging
from datetime import datetime

# ==================== parama.. ====================
REQUEST_INTERVAL_MIN = 3
REQUEST_INTERVAL_MAX = 5
MAX_RETRIES = 3
MAX_ERRORS_PER_POI = 3
MAX_REQUESTS_BEFORE_REST = 100
REST_INTERVAL = 300  # 5

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

COOKIES = [
    'WEIBOCN_FROM=1110006030; SCF=AslA_joc0x96zfUyHrhb7J2FAHHKtXWkijcx4b4EYkra1kgOT2br3xOUmtF0kyOqchockXsaO_31tenscEcyiMk.; SUB=_2A25ETuhvDeThGeFK6loS8SjPzT6IHXVnImWnrDV6PUJbktAYLW3DkW1NQ5N48SWaN4P1GVsvtwpAdawzxTSpmhxd; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WF5EC_J4w8gToJs_rQjelE95JpX5KMhUgL.FoMXeKn0eKq0Soz2dJLoIpRLxKqLBo-L1h2LxKqL1KnL1-qLxK.L1-zLBKypqP5t; SSOLoginState=1766496319; ALF=1769088319; _T_WM=55267920440; MLOGIN=1; XSRF-TOKEN=473210; M_WEIBOCN_PARAMS=fid%3D100101B2094751D36DABFF4192%26uicode%3D10000011',
    'SUB=_2AkMedLbqf8NxqwFRmv0Szmnjb490wg_EieKoKEcxJRM3HRl-yT9kqnUotRB6NfSYBQePHsNZZKggPH_9SIk72490AaSq; WEIBOCN_FROM=1110005030; MLOGIN=0; _T_WM=45917162962; XSRF-TOKEN=12e308; mweibo_short_token=682c6b1c76; M_WEIBOCN_PARAMS=fid%3D23065700428008614090000000000%26uicode%3D10000011',
]

# ==================== end ====================

# set
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weibo_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_random_delay():
    """random time"""
    return random.uniform(REQUEST_INTERVAL_MIN, REQUEST_INTERVAL_MAX)


def get_random_user_agent():
    """chose User-Agent"""
    return random.choice(USER_AGENTS)


class CookieManager:
    """Cookie"""

    def __init__(self, cookies=None):
        self.cookies = cookies or []
        self.current_index = 0

    def get_next_cookie(self):
        """next Cookie"""
        if not self.cookies:
            return None
        cookie = self.cookies[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.cookies)
        return cookie


def create_session():
    """requests"""
    session = requests.Session()
    session.headers.update({
        'Referer': 'https://m.weibo.cn/p/cardlist?',
        'User-Agent': get_random_user_agent(),
    })
    return session


def save_data(df, filename, mode='a'):
    """save CSV"""
    try:
        header = mode == 'w' or not os.path.exists(filename)
        df.to_csv(filename, index=False, mode=mode, encoding='utf-8-sig', header=header)
        return True
    except Exception as e:
        logger.error(f"error save: {e}")
        return False


def parse_weibo_data(json_data, is_first_page=False):
    """text and next page"""
    try:
        #
        if is_first_page:
            # cards[1]
            cards = json_data.get('data', {}).get('cards', [])
            if len(cards) < 2:
                return pd.DataFrame()
            data_card = cards[1]
        else:
            # cards[0]
            data_card = json_data.get('data', {}).get('cards', [{}])[0]

        #
        card_group = data_card.get('card_group', [])
        #
        if is_first_page and card_group:
            card_group = card_group[1:]

        weibos = []
        for item in card_group:
            mblog = item.get('mblog', {})
            if not mblog:
                continue

            # URL
            pics = [pic.get('url', '') for pic in mblog.get('pics', [])]

            weibos.append({
                'created_at': mblog.get('created_at', ''),
                'mid': str(mblog.get('mid', '')),
                'uid': str(mblog.get('user', {}).get('id', '')),
                'text': mblog.get('text', ''),
                'source': mblog.get('source', ''),
                'pics': json.dumps(pics, ensure_ascii=False)  #
            })

        return pd.DataFrame(weibos)
    except Exception as e:
        logger.error(f"error data: {e}")
        return pd.DataFrame()


def convert_time(time_str):
    """time trans"""
    try:
        if not time_str:
            return ''
        # : "Wed Nov 13 18:32:09 +0800 2024"
        dt = datetime.strptime(time_str, '%a %b %d %H:%M:%S %z %Y')
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return time_str


def fetch_page(session, url, cookie_manager):
    """only page"""
    retry_count = 0

    while retry_count <= MAX_RETRIES:
        try:
            cookie = cookie_manager.get_next_cookie()
            if not cookie:
                logger.error("no Cookie")
                return None

            headers = {'Cookie': cookie}

            #
            time.sleep(get_random_delay())

            response = session.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                return response
            elif response.status_code in [403, 429]:
                logger.warning(f"s: {response.status_code}")
                time.sleep(10)
            else:
                logger.warning(f"error: {response.status_code}")

        except Exception as e:
            logger.warning(f"request error: {e}")

        retry_count += 1
        time.sleep(5 * retry_count)  #

    return None


def crawl_poi(poiid, session, cookie_manager):
    """"""
    all_data = []
    page = 1

    while True:
        # URL：
        if page == 1:
            url = f'https://m.weibo.cn/api/container/getIndex?jumpfrom=weibocom&containerid=100101{poiid}'
        else:
            url = f'https://m.weibo.cn/api/container/getIndex?jumpfrom=weibocom&containerid=100101{poiid}&page={page}'

        logger.info(f" poiid={poiid}, {page}")

        response = fetch_page(session, url, cookie_manager)
        if not response:
            logger.error(f" poiid={poiid} {page}")
            break

        try:
            json_data = response.json()

            #
            if json_data.get('ok') != 1:
                logger.info(f"poiid={poiid} ")
                break

            #
            is_first_page = (page == 1)
            df = parse_weibo_data(json_data, is_first_page)

            if df.empty:
                logger.info(f"poiid={poiid} {page}")
                break

            #
            df['poiid'] = poiid
            df['created_at'] = df['created_at'].apply(convert_time)
            all_data.append(df)

            #
            page += 1

            #
            time.sleep(get_random_delay())

        except json.JSONDecodeError:
            logger.error(f"JSONerror: {response.text[:100]}")
            break
        except Exception as e:
            logger.error(f"data error: {e}")
            break

    return all_data


def main():
    """main"""
    #
    os.makedirs('data/shenzhen', exist_ok=True)

    #
    cookie_manager = CookieManager(COOKIES)
    session = create_session()

    #
    try:
        df_poiid = pd.read_csv('data/shenzhen/missing_poiid.csv', encoding='utf-8')
    except:
        df_poiid = pd.read_csv('data/shenzhen/missing_poiid.csv', encoding='gbk')

    #
    crawled_poiids = set()
    if os.path.exists('data/shenzhen/shenzhen_crawled_poiid.csv'):
        try:
            df_crawled = pd.read_csv('data/shenzhen/shenzhen_crawled_poiid.csv')
            crawled_poiids = set(df_crawled['poiid'].astype(str).tolist())
        except:
            pass

    #
    total_pois = len(df_poiid)
    processed = 0
    success = 0
    start_time = time.time()
    total_requests = 0

    #
    for idx, row in df_poiid.iterrows():
        poiid = str(row['poiid'])

        #
        if poiid in crawled_poiids:
            continue

        #
        all_data = crawl_poi(poiid, session, cookie_manager)
        processed += 1
        total_requests += len(all_data) if all_data else 0

        if all_data:
            #
            combined_df = pd.concat(all_data, ignore_index=True)
            save_data(combined_df, 'data/shenzhen/shenzhen_check-ins3.csv')

            #
            crawled_df = pd.DataFrame({'poiid': [poiid]})
            save_data(crawled_df, 'data/shenzhen/shenzhen_crawled_poiid.csv')
            crawled_poiids.add(poiid)
            success += 1

            logger.info(f" poiid={poiid}, get{len(combined_df)}")
        else:
            logger.warning(f" poiid={poiid} error")

        #
        if processed % 10 == 0:
            elapsed = time.time() - start_time
            remaining = total_pois - processed
            avg_time = elapsed / processed if processed > 0 else 0
            est_time = avg_time * remaining if remaining > 0 else 0

            logger.info(f": {processed}/{total_pois}, success: {success}")
            logger.info(f"time: {elapsed:.1f}s, remain: {est_time:.1f}s")

        #
        if processed % MAX_REQUESTS_BEFORE_REST == 0:
            logger.info(f"get{processed}个POI，relax{REST_INTERVAL}")
            time.sleep(REST_INTERVAL)

    #
    total_time = time.time() - start_time
    logger.info(f"finish!")
    logger.info(f"all: {total_pois}POI, success: {success}, not: {processed - success}")
    logger.info(f"alltime: {total_time:.1f}, avg: {total_time / processed:.1f}" if processed > 0 else "无处理")


if __name__ == "__main__":
    main()
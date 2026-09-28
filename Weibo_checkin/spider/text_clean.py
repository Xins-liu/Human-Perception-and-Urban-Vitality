import re
import pandas as pd
from html import unescape
import chardet


def detect_encoding(file_path):
    """"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  #
    result = chardet.detect(raw_data)
    return result['encoding']


def clean_weibo_text(text):
    """
    clean
    """
    if pd.isna(text):
        return ""

    # 1. HTML（&quot;、&lt;）
    text = unescape(text)

    # 2. alt（：[嘻嘻]、[色]）
    #  <img alt="[...]" ...> ，get[]
    emoticon_pattern = r'<img[^>]*alt="\[([^\]]+)\]"[^>]*>'
    text = re.sub(emoticon_pattern, r'[\1]', text)

    # 3. remove HTML（eg.<a>、<span>、<br />），
    # <br />，
    text = re.sub(r'<br\s*/?>', ' ', text)
    # remove all HTML
    text = re.sub(r'<[^>]+>', '', text)

    # 4. remove URL link
    #
    text = re.sub(r'https?://\S+', '', text)

    # 5.
    text = re.sub(r'#([^#]+)#', r'\1', text)

    # 6.
    text = re.sub(r'\s+', ' ', text).strip()

    return text


#
file_path = 'data/shenzhen/shenzhen_check-ins2.csv'
encoding = detect_encoding(file_path)
print(f": {encoding}")

#
try:
    #
    df = pd.read_csv(file_path, encoding=encoding)
except:
    try:
        #
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print("utf-8-sig")
    except:
        try:
            #
            df = pd.read_csv(file_path, encoding='gbk')
            print("gbk")
        except:
            # utf-8
            df = pd.read_csv(file_path, encoding='utf-8')
            print("utf-8")

#
print("\n:")
print(df[['text']].head(3))

#
df['text'] = df['text'].apply(clean_weibo_text)

#
print("\n:")
print(df[['text']].head(3))

#
output_file = 'data/shenzhen/shenzhenqd_cleaned2.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n {output_file} (utf-8-sig)")

#
#
print("\n l_name:")
print(df.columns.tolist())

# eg
# simplified_columns = ['created_at', 'mid', 'uid', 'text', 'source', 'poiid']
# simplified_df = df[simplified_columns]
# simplified_df.to_csv('shenzhen_check-ins_simplified.csv', index=False, encoding='utf-8-sig')
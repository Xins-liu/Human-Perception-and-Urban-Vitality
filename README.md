# Human Perception and Urban Vitality

This repository contains the code, result data, and modelling data for the analysis pipeline **"street-view visual features and human perception → urban vitality"**, using Shenzhen as the study area.

> **Data note**: The original Baidu Street View images are too large in volume, and the raw Weibo check-in texts are too numerous, to be included in full; only small samples are kept here. Apart from those two raw sources, **all aggregated result data and all data used for machine learning are fully public and transparent**.

---

## 1. Repository Contents

The repository consists of three modules, corresponding to the three steps "perception measurement → vitality measurement → relationship modelling":

| Directory / File | Description |
|---|---|
| `SVI/` | **Street View Imagery (SVI) module**: street view crawling → semantic segmentation → visual feature computation → six-dimensional human perception scoring |
| `Weibo_checkin/` | **Weibo check-in module**: POI crawling → check-in records and text crawling → text cleaning → LLM-based sentiment and urban-aspect extraction → LDA topic modelling |
| `GA-XGBoost-SHAP/` | **Modelling and interpretation module**: genetic algorithm (GA) optimisation of XGBoost hyperparameters, with SHAP explaining each feature's contribution to urban vitality |
| `spatial data/` | ArcGIS file geodatabase: road network, POIs, nighttime lights, population density and other spatial base data |

---

## 2. Directory Structure

```text
.
├── README.md
├── SVI/                                    # (1) Street View Imagery module
│   ├── svispider.py                        # Batch crawling of Baidu street view panoramas (4-direction stitching)
│   ├── Semantic_Seg.py                     # mask2former-cityscapes 19-class semantic segmentation
│   ├── VF.py                               # Visual features (green view, sky view factor, complexity, enclosure, accessibility) and aggregation
│   ├── IA.py                               # Human perception scoring (6 dimensions), zensvi + Place Pulse 2.0 pretrained model
│   └── dir/
│       ├── point_coordinate_intersect.csv  # Street view sampling points generated from the OSM road network (53,309 points)
│       ├── images/                         # Street view panoramas (not released: too large)
│       └── result/
│           ├── shenzhen_seg_all.csv            # Semantic segmentation pixel statistics (247,080 rows)
│           ├── shenzhen_seg_all_final.csv      # Segmentation statistics, filtered (245,285 rows)
│           ├── shenzhen_virtual_filtered.csv   # Intermediate visual feature table (49,416 rows)
│           ├── shenzhen_perception_scores.csv  # Perception scores by direction (49,416 rows)
│           ├── shenzhen_perception_filter.csv  # Aggregated perception scores (49,416 rows)
│           ├── shenzhen_perception_filter_final.csv # Perception scores, filtered (49,057 rows)
│           └── final_results.csv               # * Final feature table: visual features + 6-D perception (49,057 rows)
├── Weibo_checkin/                          # (2) Weibo check-in module
│   ├── LDA.py                              # Topic modelling of comments (jieba + gensim LDA + pyLDAvis)
│   ├── stopword.txt                        # Chinese stopword list
│   └── spider/
│       ├── weibo_poi.py                    # Crawl Weibo check-in POI list (poiid / name / coordinates)
│       ├── weibo_poi_detail.py             # Crawl POI details (category / address / description)
│       ├── weibo_qiandao_spider.py         # Main crawler: check-in records + text (with resume support)
│       ├── text_clean.py                   # Check-in text cleaning (HTML/emoticons/topics removed, original text kept)
│       ├── LLM_anaylsis.py                 # DeepSeek LLM: sentiment score (0-1) + urban aspect extraction
│       └── data/
│           ├── weibo_poi_shenzhen.csv                  # Raw Weibo POI table (6,200 rows, GBK encoding)
│           ├── weibo_poi_shenzhen_detail.csv           # POI details (2,930 rows)
│           ├── weibo_poi_shenzhen_cleaned_trans_wgs84.csv # POIs converted to WGS84 (2,930 rows)
│           └── shenzhen/
│               ├── shenzhen_check-ins2.csv             # Raw check-in records [sample: 500 rows]
│               ├── shenzhenqd_cleaned2.csv             # Cleaned check-in texts [sample: 500 rows]
│               ├── shenzhen_crawled_poiid.csv          # POIs already crawled (2,803 rows)
│               ├── missing_poiid.csv                   # POIs pending crawling (127 rows)
│               └── result/
│                   ├── checkin_result2.xlsx            # LLM analysis results [sample: 500 rows]
│                   ├── shenzhen_checkin_counts.csv     # * Check-in counts per POI (2,930 rows)
│                   └── shenzhen_sentiscores.csv        # * Sentiment scores per POI (2,930 rows)
├── GA-XGBoost-SHAP/                        # (3) Modelling and interpretation module
│   ├── GA-XGBoost.py                       # GA optimisation of XGBoost hyperparameters + final model training
│   ├── shap.py                             # SHAP interpretability analysis and figure export
│   ├── dataset2.csv                        # * Modelling dataset (6,551 rows x 13 variables, GBK encoding)
│   ├── urban vitality.xlsx                 # Urban vitality measurement (entropy weighting: nighttime lights / economic POIs / population / social-culture / transit)
│   └── model/
│       ├── GA-XGBoost/                     # Training artefacts: best model, best hyperparameters, metrics, training history, split indices
│       └── GA-xgboost-shap/                # Generated after running shap.py (SHAP figures and importance tables)
└── spatial data/
    └── vitality.gdb/              # ArcGIS File GDB: road network, POIs, nighttime lights, population, etc.
```

---

## 3. Data Description

**Why only part of the data is provided**

- The **original Baidu Street View images** are too large in volume and are subject to Baidu Maps' terms of service, which prohibit redistribution; they are therefore not released.
- The **raw Weibo check-in texts** are too numerous and contain large amounts of individual user text, so only 500 sample records are retained.

**What is fully released in this repository**

- The **complete code** of all three modules;
- All **aggregated results** on the SVI side: the visual feature table, the six-dimensional perception table, and the final feature table (`final_results.csv`, 49,057 points);
- All **aggregated results** on the Weibo side: check-in counts and sentiment scores at POI level (2,930 rows each);
- **All data used for machine learning**: the modelling dataset `dataset2.csv` (6,551 rows x 13 variables) and the urban vitality measurement data `urban vitality.xlsx`;
- The trained model artefacts and performance metrics (`model/GA-XGBoost/`).

Therefore, **apart from the raw imagery and raw texts, the analysis data and the modelling data of this study are fully public and the results are verifiable**.

**Variable definitions**

`dataset2.csv` (the first 12 columns are independent variables, the last column is the dependent variable; GBK encoding):

| Column | Meaning | Source |
|---|---|---|
| `Vegetation` / `Sky View Factor` | Green view index / sky view factor | SVI segmentation pixel proportions (mean over 4 directions) |
| `Visual Complexity` / `Enclosure` | Visual complexity (entropy of the 19-class proportions) / enclosure | SVI pixel statistics |
| `Spatial Accessibility` | Spatial accessibility | Road network and transit density |
| `Beautiful` / `Boring` / `Depressing` / `Lively` / `Safe` / `Wealthy` | Six-dimensional human perception | Place Pulse 2.0 pretrained ResNet50, scored per direction and then averaged |
| `Sentiment Denisity` | Sentiment density | Aggregation of Weibo check-in volume and text sentiment (spelling preserved from the original data) |
| `Urban Vitality` | **Urban vitality (dependent variable)** | Entropy-weighted composite (nighttime lights, economic POIs, population density, social-culture, transit) |

`SVI/dir/result/final_results.csv` (UTF-8-BOM, Chinese column names): `经度, 纬度, 绿视度, 天空开阔度, 复杂度, 围合感, 人行通畅度, 车辆通畅度, 通畅度, avg_more beautiful, avg_more boring, avg_more depressing, avg_livelier, avg_safer, avg_wealthier`; where `通畅度 = min(人行通畅度, 车辆通畅度)` (overall accessibility = min of pedestrian and vehicle accessibility).

`checkin_result2.xlsx` analysis fields: `analysis_sentiment_score` (0-1, with 0.5 neutral), `analysis_sentiment_category` (positive/negative/neutral), `analysis_urban_aspects` (urban aspects as JSON), `analysis_analysis_success` (whether the analysis succeeded).

**Encoding note**: `dataset2.csv` and `weibo_poi_shenzhen.csv` are GBK; `final_results.csv` is UTF-8-BOM; the rest are mostly UTF-8.

---

## 4. Environment and Dependencies

> `environment.yml` and `requirements.txt` are not provided yet (planned). The libraries actually used by the three modules are listed below.

Verified environment of this project: Python 3.9.13, pandas 2.2.3, numpy 1.26.4.

```bash
# General data processing
pip install pandas numpy scipy scikit-learn tqdm chardet openpyxl

# Module (1) SVI processing
pip install requests pillow torch transformers zensvi matplotlib

# Module (2) Weibo check-in processing
pip install requests sqlalchemy openai jieba gensim pyLDAvis

# Module (3) GA-XGBoost-SHAP
pip install xgboost scikit-learn shap matplotlib
```

**External dependencies (not pip-installable)**

- **Baidu Maps open platform AK**: required for street view crawling and coordinate conversion; please apply for your own (left blank in the code);
- **Semantic segmentation model**: a local Cityscapes version of the mask2former model directory is required (referred to as `mask2former_model_cityscapes` in the code; adjust to your actual path);
- **DeepSeek API key**: required by `LLM_anaylsis.py` to call the LLM (left blank in the code);
- **Weibo login cookies**: required to crawl check-in data; please obtain your own and fill them in (cleared in this repository).

> Note: the keys and cookies in `svispider.py`, `weibo_poi.py`, `weibo_qiandao_spider.py`, and `LLM_anaylsis.py` have all been replaced with placeholders. Please fill them in before running.

---

## 5. License

- Data license: for non-commercial academic research only; **redistribution is not permitted**.
- Note: the **street view imagery** is copyrighted by Baidu Maps and is not included in this repository; the **Weibo texts** are samples of publicly visible content, provided only to demonstrate the code pipeline, and must not be used for user profiling, individual identification, or commercial purposes.

---

## 6. Contact

- Email: 15366036911@163.com
- GitHub Issues: please open an issue in this repository for code-related problems.
- Data requests: for the full Weibo data (de-identified, aggregated level) or further details on the street view processing, please contact the email above.

---

## 7. Acknowledgements

We thank OpenStreetMap for the road network base data, the Place Pulse 2.0 and Cityscapes public datasets together with their pretrained models (mask2former, ResNet50), and open-source toolkits such as `zensvi` and `shap`. The base data for the urban vitality measurement (nighttime lights, POIs, population density, transit, etc.) come from public statistics and geospatial data platforms.

import os, re, pandas as pd, numpy as np
import jieba
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='jieba')

class Config:
    INPUT_DIR = "category_output"
    OUTPUT_DIR = "lda_results"
    STOPWORDS_FILE = "stopwords.txt"          # External stopwords file
    K_MIN, K_MAX, K_STEP = 3, 20, 1
    TOP_WORDS = 15
    REMOVE_LOCATION = True

    # Custom noise words (high-frequency words irrelevant to topics, add directly here)
    CUSTOM_NOISE = {'深圳', '全文', '展开', '网页', '链接', '查看', '更多', '原文', '转发', '评论','今天', '今日'
        , '一天', '评论', '一次'



                    }


# ------------------ Cleaning functions ------------------
def clean_text(text):
    if Config.REMOVE_LOCATION:
        text = re.sub(r'\s*[，,。.]?\s*[\u4e00-\u9fff]{2,4}·.+$', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------ Load stopwords ------------------
def load_stopwords(path):
    stop = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            stop = set(line.strip() for line in f if line.strip())
    return stop | Config.CUSTOM_NOISE

# ------------------ Tokenization and preprocessing ------------------
def tokenize(text, stopwords):
    words = jieba.lcut(text)
    return [w for w in words if w not in stopwords and len(w)>=2 and re.search(r'[\u4e00-\u9fff]', w)]

def preprocess(docs, stopwords):
    cleaned = [clean_text(d) for d in docs if pd.notna(d)]
    cleaned = [c for c in cleaned if c]
    tokens = [tokenize(c, stopwords) for c in tqdm(cleaned, desc='Tokenizing')]
    return [t for t in tokens if len(t)>=3]

# ------------------ Find optimal number of topics (save process data) ------------------
def find_optimal_k(texts, out_dir):
    dic = corpora.Dictionary(texts)
    dic.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dic.doc2bow(t) for t in texts]
    ks = list(range(Config.K_MIN, Config.K_MAX+1, Config.K_STEP))
    scores = []
    for k in ks:
        lda = models.LdaModel(corpus, id2word=dic, num_topics=k, random_state=42, passes=10, alpha='auto')
        cm = CoherenceModel(model=lda, texts=texts, dictionary=dic, coherence='c_v')
        scores.append(cm.get_coherence())
    # Save the optimization process data
    pd.DataFrame({'K': ks, 'Coherence': scores}).to_csv(
        os.path.join(out_dir, 'coherence_scores.csv'), index=False
    )
    best_idx = np.argmax(scores)
    best_k = ks[best_idx]
    final_lda = models.LdaModel(corpus, id2word=dic, num_topics=best_k, random_state=42, passes=15, alpha='auto')
    return best_k, final_lda, dic, corpus, scores[best_idx]

# ------------------ Save results and visualization ------------------
def save_results(name, lda, dic, corpus, texts, k, coh):
    out = os.path.join(Config.OUTPUT_DIR, name.replace('/', '_'))
    os.makedirs(out, exist_ok=True)

    # Model and dictionary
    lda.save(f'{out}/model.gensim')
    dic.save(f'{out}/dict.gensim')

    # Topic-word table
    topics = lda.show_topics(k, Config.TOP_WORDS, formatted=False)
    df_top = pd.DataFrame([{'Topic':t, 'Keywords':' '.join([w for w,_ in ws]),
                            'Weighted':' '.join([f'{w}({s:.4f})' for w,s in ws])} for t,ws in topics])
    df_top.to_csv(f'{out}/topics.csv', index=False, encoding='utf-8-sig')

    # Document-topic assignment
    docs = []
    for i, bow in enumerate(corpus):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        main = max(dist, key=lambda x:x[1]) if dist else (-1,0.0)
        docs.append({'Doc_ID':i, 'Topic':main[0], 'Prob':main[1], 'Dist':str(dist), 'Text':' '.join(texts[i])})
    df_doc = pd.DataFrame(docs)
    df_doc.to_csv(f'{out}/doc_topics.csv', index=False, encoding='utf-8-sig')

    # Markdown report
    md = [f"# {name}\n\n- Optimal number of topics: **{k}**\n- Coherence: **{coh:.4f}**\n\n## Topic keywords (Top {Config.TOP_WORDS})\n"]
    for t,ws in topics:
        md.append(f"### Topic {t}\n"+' | '.join([f'{w}({s:.4f})' for w,s in ws])+'\n\n')
    md.append("## Representative documents (2 per topic)\n")
    for t in range(k):
        samp = df_doc[df_doc.Topic==t].nlargest(2, 'Prob')
        md.append(f"### Topic {t}\n")
        for _,r in samp.iterrows():
            md.append(f"- {r.Text[:200]}... (probability: {r.Prob:.3f})\n")
        md.append('\n')
    with open(f'{out}/report.md', 'w', encoding='utf-8') as f: f.writelines(md)

    # pyLDAvis visualization (keep both Chinese original and forced English versions)
    try:
        vis = gensimvis.prepare(lda, corpus, dic, sort_topics=False)

        # 1. Chinese original (auto-translated by browser language)
        pyLDAvis.save_html(vis, f'{out}/vis_chinese.html')

        # 2. Forced English version (disable browser translation)
        html_str = pyLDAvis.prepared_data_to_html(vis)
        html_str = html_str.replace('</head>', '<meta name="google" content="notranslate"></head>')
        with open(f'{out}/vis_english.html', 'w', encoding='utf-8') as f:
            f.write(html_str)
    except Exception as e:
        print(f'  pyLDAvis generation failed: {e}')

# ------------------ Main process ------------------
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    stopwords = load_stopwords(Config.STOPWORDS_FILE)

    for file in os.listdir(Config.INPUT_DIR):
        if not file.endswith('.csv'): continue
        name = os.path.splitext(file)[0]
        print(f'\nProcessing category: {name}')

        # Read CSV (assume the first column is the comment text)
        df = pd.read_csv(os.path.join(Config.INPUT_DIR, file), encoding='utf-8-sig')
        docs = df.iloc[:,0].astype(str).tolist()

        # Preprocessing
        tokens = preprocess(docs, stopwords)
        if len(tokens) < 20:
            print(f'  Insufficient valid documents ({len(tokens)}), skipping')
            continue

        # Find optimal number of topics (create output directory in advance to save process data)
        out_dir = os.path.join(Config.OUTPUT_DIR, name.replace('/', '_'))
        os.makedirs(out_dir, exist_ok=True)
        k, lda, dic, corpus, coh = find_optimal_k(tokens, out_dir)
        print(f'  Optimal K={k}, Coherence={coh:.4f}')

        # Save final results
        save_results(name, lda, dic, corpus, tokens, k, coh)

    print('\nAll categories processed!')

if __name__ == '__main__':
    main()
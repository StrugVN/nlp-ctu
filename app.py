import os
import time
import joblib
from nltk.tokenize.treebank import TreebankWordDetokenizer
import streamlit as st
st.set_page_config(page_title="Search UI", layout="wide")
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import py_vncorenlp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import kenlm
from query_processing import beam_search_kenlm, load_vocab_from_file, rank_documents_by_query_enhanced, generate_progressive_suggestions
from gensim.models import Word2Vec
import re
import joblib

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["nlp"]
tf_idf_collection = db["article_tf_idf"]
article_collection = db["article"]    

# --- Load data ---

@st.cache_data
def load_tf_idf():
    print("Loading TF-IDF data from MongoDB...")
    list_tf_idf = list(tf_idf_collection.find({}))
    rows = []
    for doc in list_tf_idf:
        article_id = doc['articleId']
        tf_idf = doc.get('tf_idf', {})
        tf_idf['articleId'] = article_id
        rows.append(tf_idf)

    df = pd.DataFrame(rows)
    df.set_index('articleId', inplace=True)
    df = df.fillna(0)
    print(f"TF-IDF data loaded with shape: {df.shape}")
    return df

@st.cache_data
def load_stopwords():
    with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set(line.strip().lower() for line in f if line.strip())
    stopwords.add('sto')
    print(f"Loaded {len(stopwords)} stopwords.")
    return stopwords

@st.cache_resource
def load_segmenter():
    original_cwd = os.getcwd()
    try:
        segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=os.path.join(original_cwd, "vncorenlp"))
        os.chdir(original_cwd)
        print("VnCoreNLP segmenter initialized successfully.")
        return segmenter
    except Exception as e:
        print(f"Error initializing VnCoreNLP: {e}")
        raise e

# Load TF-IDF in background
# start_df_load = "df_tf_idf_future" in st.session_state
# if not start_df_load:
#     executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
#     df_tf_idf_future = executor.submit(load_tf_idf)

@st.cache_resource
def load_kenlm():
    return kenlm.Model("vi_model_6gramVinToken.binary")

@st.cache_resource
def load_detokenizer():
    return TreebankWordDetokenizer().detokenize

@st.cache_data
def load_vocab():
    return load_vocab_from_file("vietDict.txt")

@st.cache_resource
def load_w2v():
    return Word2Vec.load("word2vec_vi_bao_st.model")

@st.cache_resource
def load_vectorizer():
    return joblib.load('tfidf_vectorizer.joblib')

@st.cache_resource
def load_tfidf_matrix():
    return joblib.load('tfidf_matrix.joblib')

@st.cache_resource
def load_article_ids():
    return pd.read_csv('article_ids.csv')['id'].values

with st.spinner("Loading resources..."):
    stopwords       = load_stopwords()
    rdrsegmenter    = load_segmenter()
    kenMlModel      = load_kenlm()
    detokenize      = load_detokenizer()
    vocab           = load_vocab()
    word2vec_model  = load_w2v()
    vectorizer      = load_vectorizer()
    tfidf_matrix    = load_tfidf_matrix()
    article_ids     = load_article_ids()
    print("All resources loaded successfully.")
    print(len(vectorizer.get_feature_names_out().tolist()), "features in vectorizer")


# --- Search Articles ---
def search_articles_enhanced(query, top_k=10):
    # with st.spinner("Still loading TF-IDF data..."):
    #     while not df_tf_idf_future.done():
    #         time.sleep(0.1)  # Small wait to let spinner display
    #     df_tf_idf_full = df_tf_idf_future.result()

    with st.spinner("Searching articles..."):
        results, stats, weights, groups = rank_documents_by_query_enhanced(
            query=query,
            tfidf_matrix=tfidf_matrix,
            word_model=word2vec_model,
            tokenizer=rdrsegmenter,
            stopwords=stopwords,
            vectorizer=vectorizer,
            article_ids=article_ids,
            ngram_max=6,
        )

        
        result_ids = results[:top_k]
        result_articles = list(article_collection.find({
            "id": {"$in": [item[0] for item in result_ids]}
        }))
        
        id_to_article = {article["id"]: article for article in result_articles}
        sorted_articles = [id_to_article[item[0]] for item in result_ids if item[0] in id_to_article]

        print(f'\nDependency groups: {groups}')
        print(f'\nTokenized query: {weights}')

        return sorted_articles

# ---

MONGO_SEARCH = "MongoDB"
COSINE_SEARCH = "Cosine"

colA, colB = st.columns([4, 1])
with colB:
    search_mode = st.selectbox(
        "Search method",
        options=[MONGO_SEARCH, COSINE_SEARCH],
        index=1,
        key="search_mode",
        label_visibility="collapsed",
    )

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Remove default padding and center content */
    .main .block-container {
        padding-top: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .search-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 0px;
    }

    .search-input input {
        width: 400px !important;
        height: 40px;
        border-radius: 20px;
        border: 1px solid #ccc;
        padding: 0 15px;
        font-size: 16px;
    }

    .search-button button {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        font-size: 18px;
        margin-left: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- UI Layout with columns ---
st.markdown('<div class="search-container">', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

with st.form(key="search_form"):
    search_cols = st.columns([5, 1])
    with search_cols[0]:
        query = st.text_input(
            "Search box",
            placeholder="Search...",
            label_visibility="collapsed",
            disabled=st.session_state.is_processing
        )
    with search_cols[1]:
        submitted = st.form_submit_button("🔍", disabled=st.session_state.is_processing)


st.markdown("</div>", unsafe_allow_html=True)

if submitted and query.strip():
    if not st.session_state.is_processing:
        st.session_state.is_processing = True  # ⏳ Set processing state
        
        try:
            if search_mode == MONGO_SEARCH:
                st.success(f"🔎 You searched for: **{query}**")
                print(f"Using MongoDB search for query: {query}")
                searchQuery = query
                results = list(
                    article_collection.find(
                        {"$text": {"$search": query}}, {"score": {"$meta": "textScore"}}
                    )
                    .sort([("score", {"$meta": "textScore"})])
                    .limit(10)
                )
            elif search_mode == COSINE_SEARCH:
                searchQuery = query
                beamResult = beam_search_kenlm(query.lower().split(), kenMlModel)
                if beamResult:
                    searchQuery = detokenize(beamResult[0][0])

                if searchQuery != query:
                    st.warning(f"Showing result for **{searchQuery}** instead")
                else:
                    st.success(f"🔎 You searched for: **{searchQuery}**")

                print(f"Using Cosine search for query: {searchQuery}")
                results = search_articles_enhanced(searchQuery)

            st.markdown("### Search Results")
            if len(results) == 0:
                st.markdown("No results found.")
            else:
                for r in results:
                    image_url = "https://baosoctrang.org.vn" + r.get("avatarApp", "")
                    page_url = "https://baosoctrang.org.vn" + r.get("pageUrl", "")
                    col_img, col_text = st.columns([2, 5])
                    with col_img:
                        if image_url:
                            st.image(image_url, use_container_width=True)
                    with col_text:
                        st.markdown(f"""
                            <h3 style='margin-bottom: 0px;'>
                                <a href="{page_url}" style="text-decoration: none; color: black;" target="_blank">{r['title']}</a>
                            </h3>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<p style='margin-top: 2px'>{r['content'][:500]}...</p>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

        finally:
            st.session_state.is_processing = False  # ✅ Reset state

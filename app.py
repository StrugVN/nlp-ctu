import os
import time
from exceptiongroup import catch
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
from query_processing import beam_search_kenlm, load_vocab_from_file, rank_documents_by_query_enhanced, generate_vietnamese_sentences
from gensim.models import Word2Vec
import re
import joblib
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")


@st.cache_resource
def get_mongo_collections():
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client["nlp"]
    tf_idf_collection = db["article_tf_idf"]
    article_collection = db["article"]
    return tf_idf_collection, article_collection

tf_idf_collection, article_collection = get_mongo_collections()

if "forceNoSpellCorrection" not in st.session_state:
    st.session_state.forceNoSpellCorrection = False


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
        segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg", "pos", "ner", "parse"],
            save_dir=os.path.join(original_cwd, "vncorenlp")
        )
        os.chdir(original_cwd)
        print("VnCoreNLP segmenter initialized successfully.")
        return segmenter
    except Exception as e:
        print(f"Error initializing VnCoreNLP: {e}")
        raise e

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

@st.cache_resource
def load_vibert_model():
    return AutoModel.from_pretrained("vinai/phobert-base", torch_dtype="auto", cache_dir="./transformers_cache")

@st.cache_resource
def load_vibert_tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base", cache_dir="./transformers_cache")

@st.cache_resource
def load_gpt2_tokenizer():
    return AutoTokenizer.from_pretrained(r"gpt2-vietnamese-finetuned\final")

@st.cache_resource
def load_gpt2_model():
    return AutoModelForCausalLM.from_pretrained(r"gpt2-vietnamese-finetuned\final")

with st.spinner("Loading resources..."):
    stopwords        = load_stopwords()
    rdrsegmenter     = load_segmenter()
    kenMlModel       = load_kenlm()
    detokenize       = load_detokenizer()
    vocab            = load_vocab()
    word2vec_model   = load_w2v()
    vectorizer       = load_vectorizer()
    tfidf_matrix     = load_tfidf_matrix()
    article_ids      = load_article_ids()
    vibert_model     = load_vibert_model()
    vibert_tokenizer = load_vibert_tokenizer()
    tokenizer_gpt_vi = load_gpt2_tokenizer()
    model_gpt_vi     = load_gpt2_model()
    print("All resources loaded successfully.")

# --- Search logic ---
def search_articles_enhanced(query, top_k=10):
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
            bert_model=vibert_model,
            bert_tokenizer=vibert_tokenizer,
            kenlm_model=kenMlModel,
            generative_model=model_gpt_vi,
            generative_tokenizer=tokenizer_gpt_vi
        )
        result_ids = results[:top_k]
        result_articles = list(article_collection.find({
            "id": {"$in": [item[0] for item in result_ids]}
        }))
        id_to_article = {article["id"]: article for article in result_articles}
        sorted_articles = [id_to_article[item[0]] for item in result_ids if item[0] in id_to_article]
        print(f'\nTokenized query: {weights}')
        return sorted_articles
    

def auto_complete(query):
    sentence_list = generate_vietnamese_sentences(query, model_gpt_vi, tokenizer_gpt_vi, rdrsegmenter)

    sentence_list = sorted(sentence_list, key=lambda x: kenMlModel.score(x), reverse=True)

    return sentence_list

# --- Constants ---
MONGO_SEARCH = "MongoDB"
COSINE_SEARCH = "Cosine"

# --- CSS (unchanged) ---
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1rem; padding-left: 2rem; padding-right: 2rem; }
    .search-container { display: flex; align-items: center; justify-content: center; margin-top: 0px; }
    .search-input input { width: 400px !important; height: 40px; border-radius: 20px; border: 1px solid #ccc; padding: 0 15px; font-size: 16px; }
    .search-button button { border-radius: 50%; width: 40px; height: 40px; font-size: 18px; margin-left: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
    <style>
    .search-instead-btn {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.3em 0.8em !important;
        font-size: 0.9em !important;
        font-weight: normal !important;
        text-align: left !important;
        margin-left: 1.5em !important;  /* indent */
    }
    .search-instead-btn:hover {
        background-color: #ffe8a1 !important;
    }
    </style>
    """, unsafe_allow_html=True
)



# --- Session defaults ---
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_mode" not in st.session_state:
    st.session_state.last_mode = COSINE_SEARCH
if "queued_query" not in st.session_state:
    st.session_state.queued_query = None

# If something queued a search on the previous run, start it now
if st.session_state.queued_query:
    st.session_state.last_query = st.session_state.queued_query
    st.session_state.is_processing = True
    st.session_state.queued_query = None


# =========================
#        HEADER (static)
# =========================
st.markdown('<div class="search-container">', unsafe_allow_html=True)

# Row 1: right-aligned method selector (unchanged look)
colA, colB = st.columns([4, 1])
with colB:
    st.session_state.last_mode = st.selectbox(
        "Search method",
        options=[MONGO_SEARCH, COSINE_SEARCH],
        index=1 if st.session_state.last_mode == COSINE_SEARCH else 0,
        key="search_mode",
        label_visibility="collapsed",
    )

# Row 2: input + button (ENTER submits too)
with st.form("search_form", clear_on_submit=False):
    cols = st.columns([5, 1])
    with cols[0]:
        # keep value persistent so users can edit and re-search
        query_input = st.text_input(
            "Search box",
            value=st.session_state.last_query,
            placeholder="Search.",
            label_visibility="collapsed",
            key="query_input_text",
            #disabled=st.session_state.is_processing,
        )
    with cols[1]:
        do_search = st.form_submit_button(
            "🔍"
        )  # Enter now triggers this too

suggestions = auto_complete(query_input) if query_input else []


st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Clicking the button updates state; header stays visible, body updates
if do_search and query_input.strip():
    st.session_state.last_query = query_input.strip()
    st.session_state.last_mode = st.session_state.get("search_mode", COSINE_SEARCH)
    st.session_state.is_processing = True

# =========================
#          BODY 
# =========================
body = st.empty()

def _trigger_search_with_original(q):
    print('Triggering search with original query:', q)
    st.session_state.forceNoSpellCorrection = True
    st.session_state.last_query = q
    st.session_state.is_processing = True            # enter processing branch next run


def render_results(results, shown_query, original_query):
    with body.container():
        # Header lines
        st.success(f"🔎 Results for **{shown_query}**")

        if shown_query != original_query:
            st.button(
                f"_Search instead for: **{original_query}**_",
                type="tertiary",
                key=f"search_instead_{original_query}",
                icon="❗",
                on_click=_trigger_search_with_original,
                args=(original_query,),
            )    


        for r in results:
            image_url = "https://baosoctrang.org.vn" + r.get("avatarApp", "")
            page_url  = "https://baosoctrang.org.vn" + r.get("pageUrl", "")
            col_img, col_text = st.columns([2, 5])
            with col_img:
                if image_url:
                    st.image(image_url, use_container_width=True)
            with col_text:
                st.markdown(
                    f"""
                    <h3 style='margin-bottom: 0px;'>
                        <a href="{page_url}" style="text-decoration: none; color: black;" target="_blank">{r['title']}</a>
                    </h3>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(f"<p style='margin-top: 2px'>{r['content'][:500]}...</p>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)


print("=====================================")
print("is_processing:", st.session_state.is_processing)
print("last_query:", st.session_state.last_query)
print("last_mode:", st.session_state.last_mode)
print("queued_query:", st.session_state.queued_query)
print("forceNoSpellCorrection:", st.session_state.forceNoSpellCorrection)
print("=====================================")

# If we’re processing, show spinner + run the search, all inside BODY only
if st.session_state.is_processing:
    print(">>>>>>>>>>>>>>>>>>>>>>>>Processing search<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    
    original_query = st.session_state.last_query
    searchQuery = original_query
    beamResult = beam_search_kenlm(searchQuery.lower().split(), kenMlModel)
    if not st.session_state.forceNoSpellCorrection and beamResult:
        searchQuery = detokenize(beamResult[0][0])
    st.session_state.forceNoSpellCorrection = False
    
    try:
        print(f'Test suggestion: ', auto_complete(searchQuery))
    except Exception as e:
        print(f"Error generating suggestions: {e}")

    with body.container():
        # Run the search according to mode
        if st.session_state.last_mode == MONGO_SEARCH:
            st.info(f"🔎 You searched for: **{original_query}**")
            
            with st.spinner("Searching articles..."):
                results = list(
                    article_collection.find(
                        {"$text": {"$search": original_query}},
                        {"score": {"$meta": "textScore"}}
                    )
                    .sort([("score", {"$meta": "textScore"})])
                    .limit(10)
                )

            render_results(results, original_query, original_query)
        else:
            st.info(f"🔎 Searching for: **{searchQuery}**")
            results = search_articles_enhanced(searchQuery)
            
            render_results(results, searchQuery, original_query)

    # After work, replace spinner with results
    st.session_state.is_processing = False
else:
    with body.container():
        st.markdown("<h2>Welcome to the Search UI</h2>", unsafe_allow_html=True)
        st.markdown("Type your query above and click the search button to find articles.")

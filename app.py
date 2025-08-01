import os
import time
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
from fix_tonal import beam_search_kenlm, load_vocab_from_file, generate_progressive_suggestions

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
        segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.join(original_cwd, "vncorenlp"))
        os.chdir(original_cwd)
        print("VnCoreNLP segmenter initialized successfully.")
        return segmenter
    except Exception as e:
        print(f"Error initializing VnCoreNLP: {e}")
        raise e

# Load TF-IDF in background
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
df_tf_idf_future = executor.submit(load_tf_idf)

with st.spinner("Loading resources..."):
    stopwords = load_stopwords()
    rdrsegmenter = load_segmenter()
    kenMlModel = kenlm.Model("vi_model_6gramVinToken.binary")
    detokenize = TreebankWordDetokenizer().detokenize
    vocab = load_vocab_from_file("vietDict.txt")

# ---
def rank_documents_by_query(query, tf_idf, tokenizer, stopwords):
    segmented = tokenizer.word_segment(query)
    query_tokens = []
    for sentence in segmented:
        words = sentence.split()
        words = [w.replace("_", " ") for w in words]
        words = [w.lower() for w in words if w.lower() not in stopwords]
        query_tokens.extend(words)

    word_list = tf_idf.columns
    query_vector = np.zeros(len(word_list))

    word_counts = {word: query_tokens.count(word) for word in set(query_tokens)}

    total_terms = sum(word_counts.values())
    if total_terms == 0:
        return []

    for i, term in enumerate(word_list):
        if term in word_counts:
            query_vector[i] = word_counts[term] / total_terms 

    cosin_sim = cosine_similarity([query_vector], tf_idf.values)[0]

    article_ids = tf_idf.index.tolist()
    ranked = sorted(zip(article_ids, cosin_sim), key=lambda x: x[1], reverse=True)
    
    return ranked

def search_articles(query):
    with st.spinner("Still loading TF-IDF data..."):
        while not df_tf_idf_future.done():
            time.sleep(0.1)  # Small wait to let spinner display
        df_tf_idf_full = df_tf_idf_future.result()

    with st.spinner("Searching articles..."):
        results = rank_documents_by_query(query, df_tf_idf_full, rdrsegmenter, stopwords)
        result_ids = results[:10]
        result_articles = list(article_collection.find({"id": {"$in": [item[0] for item in result_ids]}}))
        return result_articles
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

with st.form(key="search_form"):
    search_cols = st.columns([5, 1])
    with search_cols[0]:
        query = st.text_input(
            "Search box",
            placeholder="Search...",
            label_visibility="collapsed"
        )
    with search_cols[1]:
        submitted = st.form_submit_button("🔍")


st.markdown("</div>", unsafe_allow_html=True)

if submitted and query.strip():
    if query.strip():
        if search_mode == MONGO_SEARCH:
            st.success(f"🔎 You searched for: **{query}**")
            print(f"Using MongoDB search for query: {query}")
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
                st.warning(f"Did you mean: **{searchQuery}**?")
            else:
                st.success(f"🔎 You searched for: **{searchQuery}**")

            print(f"Using Cosine search for query: {searchQuery}")

            results = search_articles(searchQuery)

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
    else:
        st.warning("Please enter something!")

    # Reset trigger
    st.session_state.submitted = False

import os
from pathlib import Path
from functools import lru_cache
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from pymongo import MongoClient
from dotenv import load_dotenv

# Heavy libs and ML imports
import pandas as pd
import numpy as np
import joblib
import kenlm
from nltk.tokenize.treebank import TreebankWordDetokenizer
import py_vncorenlp
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from query_processing import (
    beam_search_kenlm,
    load_vocab_from_file,
    rank_documents_by_query_enhanced,
    generate_vietnamese_sentences,
)

# --- Path setup ---
BASE_DIR = Path(__file__).resolve().parent
def P(*parts: str) -> str:
    return str(BASE_DIR.joinpath(*parts))

# Flask setup
load_dotenv()
app = Flask(
    __name__,
    template_folder=P("templates"),
    static_folder=P("static")
)
app.secret_key = os.getenv("FLASK_SECRET", "change-me")

# --- Mongo ---
@lru_cache(maxsize=1)
def get_mongo_collections():
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client["nlp"]
    tf_idf_collection = db["article_tf_idf"]
    article_collection = db["article"]
    return tf_idf_collection, article_collection

tf_idf_collection, article_collection = get_mongo_collections()

# --- Load resources ---
@lru_cache(maxsize=1)
def load_stopwords():
    print(f"Loading stopwords from {P('vietnamese-stopwords.txt')}")
    with open(P('vietnamese-stopwords.txt'), 'r', encoding='utf-8') as f:
        stopwords = set(line.strip().lower() for line in f if line.strip())
    stopwords.add('sto')
    print(f"Loaded {len(stopwords)} stopwords.")
    return stopwords

@lru_cache(maxsize=1)
def load_segmenter():
    original_cwd = os.getcwd()
    try:
        save_path = os.path.join(original_cwd, "vncorenlp")
        print(f"Initializing VnCoreNLP segmenter... {save_path}")
        segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg", "pos", "ner", "parse"],
            save_dir=save_path
        )
        os.chdir(original_cwd)
        print("VnCoreNLP segmenter initialized successfully.")
        return segmenter
    except Exception as e:
        print(f"Error initializing VnCoreNLP: {e}")
        raise e

@lru_cache(maxsize=1)
def load_kenlm_model():
    model = kenlm.Model(P("vi_model_6gramVinToken.binary"))
    print("Loaded KenLM model successfully.")
    return model

@lru_cache(maxsize=1)
def load_detokenizer():
    return TreebankWordDetokenizer().detokenize

@lru_cache(maxsize=1)
def load_vocab():
    vocab = load_vocab_from_file(P("vietDict.txt"))
    print(f"Loaded vocabulary with {len(vocab)} entries.")
    return vocab

@lru_cache(maxsize=1)
def load_w2v():
    model = Word2Vec.load(P("word2vec_vi_bao_st.model"))
    print("Loaded Word2Vec model successfully.")
    return model

@lru_cache(maxsize=1)
def load_vectorizer():
    model = joblib.load(P('tfidf_vectorizer.joblib'))
    print("Loaded TF-IDF vectorizer successfully.")
    return model

@lru_cache(maxsize=1)
def load_tfidf_matrix():
    model = joblib.load(P('tfidf_matrix.joblib'))
    print("Loaded TF-IDF matrix successfully.")
    return model

@lru_cache(maxsize=1)
def load_article_ids():
    ids = pd.read_csv(P('article_ids.csv'))['id'].values
    print(f"Loaded article IDs with {len(ids)} entries.")
    return ids

@lru_cache(maxsize=1)
def load_vibert_model():
    model = AutoModel.from_pretrained("vinai/phobert-base", torch_dtype="auto", cache_dir=P("transformers_cache"))
    print("Loaded ViBERT model successfully.")
    return model

@lru_cache(maxsize=1)
def load_vibert_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", cache_dir=P("transformers_cache"))
    print("Loaded ViBERT tokenizer successfully.")
    return tokenizer

@lru_cache(maxsize=1)
def load_gpt2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(P("gpt2-vietnamese-finetuned", "final"))
    print("Loaded GPT-2 tokenizer successfully.")
    return tokenizer

@lru_cache(maxsize=1)
def load_gpt2_model():
    model = AutoModelForCausalLM.from_pretrained(P("gpt2-vietnamese-finetuned", "final"))
    print("Loaded GPT-2 model successfully.")
    return model

# --- Preload all resources ---
def warm_resources():
    _ = (
        load_stopwords(),
        load_segmenter(),
        load_kenlm_model(),
        load_detokenizer(),
        load_vocab(),
        load_w2v(),
        load_vectorizer(),
        load_tfidf_matrix(),
        load_article_ids(),
        load_vibert_model(),
        load_vibert_tokenizer(),
        load_gpt2_tokenizer(),
        load_gpt2_model(),
    )
    print("All resources preloaded.")

warm_resources()

# --- Business logic ---
def search_articles_enhanced(query, top_k=10):
    stopwords        = load_stopwords()
    kenMlModel       = load_kenlm_model()
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
    rdrsegmenter     = load_segmenter()

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
    sorted_articles = [id_to_article.get(item[0]) for item in result_ids if item[0] in id_to_article]
    return sorted_articles

def autocomplete(query):
    if not query:
        return []
    kenMlModel = load_kenlm_model()
    rdrsegmenter = load_segmenter()
    tokenizer_gpt_vi = load_gpt2_tokenizer()
    model_gpt_vi = load_gpt2_model()

    sentence_list = generate_vietnamese_sentences(query, model_gpt_vi, tokenizer_gpt_vi, rdrsegmenter)
    sentence_list = sorted(sentence_list, key=lambda x: kenMlModel.score(x), reverse=True)
    return sentence_list[:5]

# --- Routes ---
@app.before_request
def ensure_session_defaults():
    session.setdefault("last_query", "")
    session.setdefault("last_mode", "Cosine")
    session.setdefault("forceNoSpellCorrection", False)

@app.route("/", methods=["GET"])
def home():
    q = session.get("last_query", "")
    mode = session.get("last_mode", "Cosine")
    suggestions = autocomplete(q) if q else []
    return render_template("index.html", last_query=q, last_mode=mode, suggestions=suggestions)

@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        q = request.form.get("q", "").strip()
        mode = request.form.get("mode", "Cosine")
    else:
        q = request.args.get("q", "").strip()
        mode = request.args.get("mode", "Cosine")

    if not q:
        return redirect(url_for("home"))

    session["last_query"] = q
    session["last_mode"] = mode

    kenMlModel = load_kenlm_model()
    detokenize = load_detokenizer()

    original_query = q
    search_query = q

    if not session.get("forceNoSpellCorrection", False):
        try:
            beamResult = beam_search_kenlm(search_query.lower().split(), kenMlModel)
            if beamResult:
                search_query = detokenize(beamResult[0][0])
        except Exception:
            pass
    session["forceNoSpellCorrection"] = False

    if mode == "MongoDB":
        results = list(
            article_collection.find(
                {"$text": {"$search": original_query}},
                {"score": {"$meta": "textScore"}}
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(10)
        )
        shown_query = original_query
    else:
        results = search_articles_enhanced(search_query, top_k=10)
        shown_query = search_query

    suggestions = autocomplete(search_query)
    return render_template(
        "results.html",
        results=results,
        original_query=original_query,
        shown_query=shown_query,
        last_mode=mode,
        suggestions=suggestions
    )

@app.route("/search_instead")
def search_instead():
    q = request.args.get("q", "").strip()
    if q:
        session["forceNoSpellCorrection"] = True
        session["last_query"] = q
        return redirect(url_for("search", q=q, mode=session.get("last_mode", "Cosine")))
    return redirect(url_for("home"))

@app.get("/api/autocomplete")
def api_autocomplete():
    q = request.args.get("q", "").strip()
    return jsonify(autocomplete(q))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

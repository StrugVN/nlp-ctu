import re
from functools import lru_cache
from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
from itertools import product
from collections import defaultdict
from itertools import combinations
import torch
from heapq import nlargest

def remove_vn_accent(word):
    word = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', word)
    word = re.sub('[éèẻẽẹêếềểễệ]', 'e', word)
    word = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', word)
    word = re.sub('[íìỉĩị]', 'i', word)
    word = re.sub('[úùủũụưứừửữự]', 'u', word)
    word = re.sub('[ýỳỷỹỵ]', 'y', word)
    word = re.sub('đ', 'd', word)
    return word

def build_accent_lookup(filepath='all-vietnamese-syllables.txt'):
    lookup = defaultdict(set)
    with open(filepath, encoding='utf-8') as f:
        for w in f.read().splitlines():
            key = remove_vn_accent(w.lower())
            lookup[key].add(w.lower())
    return lookup

ACCENT_LOOKUP = build_accent_lookup()

@lru_cache(maxsize=10000)
def gen_accents_word_cached(word):
    key = remove_vn_accent(word.lower())
    return ACCENT_LOOKUP.get(key, {word})

def beam_search_kenlm(words, model, k=3, max_len=None, force=False):
    if not needs_diacritic_restoration(" ".join(words)) and not force:
        return []

    variants = [gen_accents_word_cached(w) for w in words]

    sequences = [([], 0.0, "")]

    for word_options in variants:
        new_sequences = []
        for tokens, cum_lp, text in sequences:
            prefix_lp = model.score(text, bos=False, eos=False) if text else 0.0

            for w in word_options:
                new_text = (text + " " + w).strip()
                full_lp = model.score(new_text, bos=False, eos=False)
                inc_lp = full_lp - prefix_lp  
                new_tokens = tokens + [w]

                if max_len and len(new_tokens) > max_len:
                    continue

                new_sequences.append((new_tokens, cum_lp + inc_lp, new_text))

        sequences = nlargest(k, new_sequences, key=lambda x: x[1])
        if not sequences:
            break

    return sequences

VI_DIACRITIC_PATTERN = re.compile(r"[áàảãạăắằẳẵặâấầẩẫậ"
                                  r"éèẻẽẹêếềểễệ"
                                  r"íìỉĩị"
                                  r"óòỏõọôốồổỗộơớờởỡợ"
                                  r"úùủũụưứừửữự"
                                  r"ýỳỷỹỵ"
                                  r"đ]", re.IGNORECASE)

def needs_diacritic_restoration(text):
    """Returns True if text lacks Vietnamese tone marks and restoration is likely needed."""
    return not bool(VI_DIACRITIC_PATTERN.search(text))

def load_vocab_from_file(path='vietDict.txt'):
    with open(path, encoding='utf-8') as f:
        return [w.strip() for w in f if w.strip() and w.strip() not in ('<s>', '</s>', '<unk>')]


def generate_vietnamese_sentences(prompt, model, tokenizer, rdrsegmenter, kenMlModel, detokenize, device='cpu',  do_sample=True, top_p=0.9, top_k=50, temperature=0.8):
    prompt = prompt.strip()

    beamResult = beam_search_kenlm(prompt.lower().split(), kenMlModel)
    if beamResult:
        prompt = detokenize(beamResult[0][0])
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    max_new_tokens = prompt.count(' ') * 2

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max(max_new_tokens, 3),
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=10
        )

    prompt_segmented = rdrsegmenter.word_segment(prompt)[0].split()
    result = []
    for output_id in output_ids:
        output_text = tokenizer.decode(output_id, skip_special_tokens=True)
        output_text = re.sub(r'\s+', ' ', output_text).strip()
        
        segmented_text = rdrsegmenter.word_segment(output_text)[0].split()

        for i in range(0, 3):
            segmented_text_cropped = segmented_text[:len(prompt_segmented) + i]

            if segmented_text_cropped != prompt_segmented and segmented_text_cropped not in result:
                result.append(segmented_text_cropped)

    max_len = len(prompt_segmented) + 2
    result_sentences = [
        ' '.join(item[:max_len]).replace('_', ' ')
        for item in result
    ]

    result_sentences.sort(key=lambda x: kenMlModel.score(x) / len(x.split()), reverse=True)

    return result_sentences

def expand_query_enhanced(token, model, topn=5, similarity_threshold=0.5):
    expanded_tokens = [(token, 1.0)]
    if token in model.wv:
        similar_words = model.wv.most_similar(token, topn=topn)
        for word, similarity in similar_words:
            if similarity > similarity_threshold:
                clean_word = word.replace('_', ' ')
                expanded_tokens.append((clean_word, similarity))
    return expanded_tokens


def should_expand_token(token, stopwords, min_length=3):
    return not (token.lower() in stopwords or len(token) < min_length or token.isnumeric())

def expand_query_enhanced(token, model, topn=5, similarity_threshold=0.5):
    expanded_tokens = [(token, 1.0)]
    if token in model.wv:
        similar_words = model.wv.most_similar(token, topn=topn)
        for word, similarity in similar_words:
            if similarity > similarity_threshold:
                clean_word = word.replace('_', ' ')
                expanded_tokens.append((clean_word, similarity))
    return expanded_tokens


def should_expand_token(token, stopwords, min_length=3):
    return not (token.lower() in stopwords or len(token) < min_length or token.isnumeric())


def extract_subject_tokens(annotations):
    """
    Collect noun tokens from the root subtree, sorted by their sentence index.
    """
    subject_indices = set()
    index_to_entry = {entry["index"]: entry for entry in annotations}
    graph = {}

    for entry in annotations:
        graph.setdefault(entry["head"], []).append(entry["index"])

    root_entry = next((e for e in annotations if e["depLabel"] == "root"), None)
    if not root_entry:
        return []

    def collect(index):
        entry = index_to_entry.get(index)
        if entry and entry["posTag"].startswith("N"):
            subject_indices.add(index)
        for child in graph.get(index, []):
            collect(child)

    collect(root_entry["index"])

    # Sort indices and return corresponding wordForms
    sorted_indices = sorted(subject_indices)
    return [index_to_entry[i]["wordForm"] for i in sorted_indices]


def extract_dependency_chains(annotations, subject_tokens, stopwords, max_len=6):
    """
    For each subject token, collect all subchains from its dependency subtree,
    remove stopwords, and deduplicate final result.
    """
    index_to_entry = {entry["index"]: entry for entry in annotations}
    token_to_index = {entry["wordForm"]: entry["index"] for entry in annotations}
    graph = {}

    for entry in annotations:
        graph.setdefault(entry["head"], []).append(entry["index"])

    def dfs(index, visited):
        if index in visited:
            return
        visited.add(index)
        for child in graph.get(index, []):
            dfs(child, visited)

    final_chains = set()

    for token in subject_tokens:
        root_idx = token_to_index.get(token)
        if not root_idx:
            continue

        visited = set()
        dfs(root_idx, visited)
        sorted_indices = sorted(visited)
        sorted_tokens = [index_to_entry[i]["wordForm"] for i in sorted_indices]

        n = len(sorted_tokens)
        for i in range(n):
            for j in range(i + 1, min(i + max_len, n) + 1):
                chain = sorted_tokens[i:j]
                # Remove stopwords
                clean_chain = [tok for tok in chain if tok.lower() not in stopwords]
                if clean_chain:
                    final_chains.add(tuple(clean_chain))

    return [list(chain) for chain in final_chains]

def encode_sentence(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding


def similarity_viBERT(sent1, sent2, model, tokenizer):
    vec1 = encode_sentence(sent1, model, tokenizer).numpy() 
    vec2 = encode_sentence(sent2, model, tokenizer).numpy()
    
    sim = cosine_similarity(vec1, vec2)[0][0]
    return sim

def rank_documents_by_query_enhanced(query,
                                     tfidf_matrix,
                                     word_model,
                                     tokenizer,
                                     stopwords,
                                     vectorizer,
                                     article_ids,
                                     base_expansion_weight=0.3,
                                     adaptive_expansion=True,
                                     similarity_threshold=0.7,
                                     ngram_max=6,
                                     max_ngrams=20,
                                     bert_model=None,
                                     bert_tokenizer=None,
                                     kenlm_model=None,
                                     generative_model=None,
                                     generative_tokenizer=None
                                     ):

    segmented = tokenizer.word_segment(query)[0]

    print(f'\nSegmented query: {segmented}')

    query_tokens = []
    for token in segmented.split(' '):
        if token.replace("_", " ").lower() not in stopwords and len(token) > 1:
            query_tokens.append(token)

    annotated = tokenizer.annotate_text(query)[0]

    # subject_tokens_list = [t.lower().replace(" ", "_") for t in extract_subject_tokens(annotated)]
    # subject_tokens = set(subject_tokens_list)

    query_tokens_str = " ".join(query_tokens).replace("_", " ")

    dependency_phrases = extract_dependency_chains(annotated, query_tokens, stopwords, max_len=ngram_max)

    print('\nannotated:', annotated)
    print('\nsubject_tokens:', query_tokens) ####
    print('\ndependency_phrases:', dependency_phrases)

    if not dependency_phrases:
        return []

    original_query_length = len(dependency_phrases)
    if adaptive_expansion:
        if original_query_length <= 2:
            expansion_weight = base_expansion_weight * 1.5
        elif original_query_length >= 6:
            expansion_weight = base_expansion_weight * 0.5
        else:
            expansion_weight = base_expansion_weight
    else:
        expansion_weight = base_expansion_weight

    word_counts = {}
    expansion_stats = {"original_terms": 0, "expanded_terms": 0}
    phrase_weights = {}  

    for phrase in dependency_phrases:
        phrase_weight = 1.0

        if bert_model and bert_tokenizer:
            phrase_str = " ".join(phrase).replace("_", " ")
            phrase_weight = similarity_viBERT(query_tokens_str, phrase_str, bert_model, bert_tokenizer)**0.5
        else:
            print("BERT model or tokenizer not provided, using default weight of 1.0 for phrases.")

        phrase_key = " ".join(phrase)
        phrase_weights[phrase_key] = phrase_weight
        
        print(f'Processing phrase: {phrase} with weight: {phrase_weight:.4f}')

        expanded_phrase = []
        for token in phrase:
            token_clean = token.replace("_", " ").lower()
            options = [(token_clean, phrase_weight)]            
            if should_expand_token(token_clean, stopwords):
                expanded = expand_query_enhanced(
                    token_clean, word_model, topn=5, similarity_threshold=similarity_threshold
                )
                for exp_token, sim in expanded[1:]:
                    if exp_token != token_clean and exp_token not in stopwords:
                        weight = sim * expansion_weight * phrase_weight
                        options.append((exp_token, weight))
                        expansion_stats["expanded_terms"] += 1
            expanded_phrase.append(options)
            expansion_stats["original_terms"] += 1

        for combo in product(*expanded_phrase):
            tokens, weights = zip(*combo)
            tokens_clean = [t.replace(" ", "_") for t in tokens]
            gram = " ".join(tokens_clean)

            if not any(subj in gram for subj in query_tokens):
                continue

            weight = sum(weights) / len(weights)
            word_counts[gram] = word_counts.get(gram, 0) + weight

    if len(word_counts) > max_ngrams:
        word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_ngrams])

    total_weight = sum(word_counts.values())
    if total_weight == 0:
        return []

    feature_names = vectorizer.get_feature_names_out().tolist()
    col_index = {term.lower(): idx for idx, term in enumerate(feature_names)}
    query_vector = np.zeros(len(feature_names))

    for token, weight in word_counts.items():
        idx = col_index.get(token)
        if idx is not None:
            query_vector[idx] = weight / total_weight

    cosine_sim = cosine_similarity([query_vector], tfidf_matrix)[0]
    ranked = sorted(zip(article_ids, cosine_sim), key=lambda x: x[1], reverse=True)

    top_score = ranked[0][1]
    min_similarity = max(0.05, top_score * 0.2)

    ranked = [(doc_id, sim) for doc_id, sim in ranked if sim > min_similarity]

    return ranked, phrase_weights, word_counts, dependency_phrases

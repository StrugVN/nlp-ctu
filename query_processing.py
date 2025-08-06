import re
from functools import lru_cache
from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
from itertools import product

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

def beam_search_kenlm(words, model, k=3, force=False):
    if not needs_diacritic_restoration(" ".join(words)) and not force:
        return []

    variants = [list(gen_accents_word_cached(w)) for w in words]
    sequences = [([], 0.0)]  

    for word_options in variants:
        new_sequences = []
        for seq, score in sequences:
            prefix = " ".join(seq[-2:]) if len(seq) >= 2 else " ".join(seq)
            for word in word_options:
                full = f"{prefix} {word}".strip()
                new_score = model.score(full, bos=False, eos=False)
                new_sequences.append((seq + [word], score + new_score))
        new_sequences.sort(key=lambda x: x[1], reverse=True)
        sequences = new_sequences[:k]

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


def generate_progressive_suggestions(model, prefix_words, vocab, top_k=5):
    """
    Generate top-k suggestions with increasing length (each suggestion longer than the previous by 1 token).
    """
    from heapq import nlargest
    detokenize = TreebankWordDetokenizer().detokenize

    suggestions = []
    current_beam = [(0.0, prefix_words[:])]  

    for target_len in range(len(prefix_words) + 1, len(prefix_words) + top_k + 1):
        next_beam = []
        for score, tokens in current_beam:
            for word in vocab:
                full_seq = tokens + [word]
                full_str = " ".join(full_seq[-3:])  
                new_score = score + model.score(full_str, bos=False, eos=False)
                next_beam.append((new_score, full_seq))

        best = nlargest(1, [s for s in next_beam if len(s[1]) == target_len], key=lambda x: x[0])
        if best:
            suggestions.append(best[0])
            current_beam = best
        else:
            break 

    return [(detokenize(tokens), score) for score, tokens in suggestions]

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

def filter_redundant_ngrams(word_counts, coverage_threshold=0.8):
    """Remove n-grams that are largely covered by other n-grams"""
    sorted_ngrams = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    filtered = {}
    covered_words = set()
    
    for ngram, weight in sorted_ngrams:
        words = set(ngram.split())
        
        # Calculate how much this n-gram overlaps with already selected ones
        overlap_ratio = len(words.intersection(covered_words)) / len(words) if words else 1
        
        # Keep if it adds significant new information or is highly weighted
        if overlap_ratio < coverage_threshold or len(filtered) < 3:
            filtered[ngram] = weight
            covered_words.update(words)
    
    return filtered

from itertools import product
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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


def extract_dependency_chains(annotations, max_len=6):
    graph = {}
    index_to_token = {}
    for entry in annotations:
        idx = entry["index"]
        word = entry["wordForm"]
        head = entry["head"]
        index_to_token[idx] = word
        graph.setdefault(head, []).append(idx)

    chains = []

    def dfs(path):
        current = path[-1]
        if len(path) > max_len:
            return
        chains.append([index_to_token[i] for i in path if i in index_to_token])
        for child in graph.get(current, []):
            if child not in path:
                dfs(path + [child])

    for idx in sorted(index_to_token.keys()):
        dfs([idx])

    return chains


def extract_subject_tokens(annotations):
    subject_tokens = set()
    root_entry = next((e for e in annotations if e["depLabel"] == "root"), None)
    if not root_entry:
        return subject_tokens
    root_idx = root_entry["index"]
    subject_tokens.add(root_entry["wordForm"])
    for entry in annotations:
        if entry["head"] == root_idx and entry["posTag"].startswith("N"):
            subject_tokens.add(entry["wordForm"])
    return subject_tokens


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
                                     max_ngrams=20):

    annotated = tokenizer.annotate_text(query)[0]
    raw_dependency_phrases = extract_dependency_chains(annotated, max_len=ngram_max)
    subject_tokens = {t.lower().replace(" ", "_") for t in extract_subject_tokens(annotated)}

    print('annotated:', annotated)
    print('subject_tokens:', subject_tokens)

    dependency_phrases = []
    for phrase in raw_dependency_phrases:
        clean_phrase = [token for token in phrase if token.lower() not in stopwords]
        if clean_phrase:
            dependency_phrases.append(clean_phrase)

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

    for phrase in dependency_phrases:
        expanded_phrase = []
        for token in phrase:
            token_clean = token.replace("_", " ").lower()
            options = [(token_clean, 1.0)]
            if should_expand_token(token_clean, stopwords):
                expanded = expand_query_enhanced(token_clean, word_model, topn=5, similarity_threshold=similarity_threshold)
                for exp_token, sim in expanded[1:]:
                    if exp_token != token_clean and exp_token not in stopwords:
                        options.append((exp_token, sim * expansion_weight))
                        expansion_stats['expanded_terms'] += 1
            expanded_phrase.append(options)
            expansion_stats['original_terms'] += 1

        for combo in product(*expanded_phrase):
            tokens, weights = zip(*combo)
            tokens_clean = [t.replace(" ", "_") for t in tokens]
            gram = " ".join(tokens_clean)

            if not any(subj in gram for subj in subject_tokens):
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

    return ranked, expansion_stats, word_counts, dependency_phrases

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
    return not(token.lower() in stopwords 
               or len(token) < min_length 
               or token.isnumeric()
            )

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
                                     ngram_max=3,
                                     max_ngrams=20,  # New parameter to limit n-grams
                                    ):
    
    # Step 1: Tokenize and clean query
    segmented = tokenizer.word_segment(query)
    query_tokens = []
    for sentence in segmented:
        words = sentence.split()
        words = [w.replace("_", " ") for w in words]
        words = [w.lower() for w in words if w.lower() not in stopwords]
        query_tokens.extend(words)

    if not query_tokens:
        return []

    original_query_length = len(query_tokens)

    # Step 3: Adjust expansion weight
    if adaptive_expansion:
        if len(query_tokens) <= 2:
            expansion_weight = base_expansion_weight * 1.5
        elif len(query_tokens) >= 6:
            expansion_weight = base_expansion_weight * 0.5
        else:
            expansion_weight = base_expansion_weight
    else:
        expansion_weight = base_expansion_weight

    # Step 4: Expand tokens
    token_options = []
    expansion_stats = {'original_terms': len(query_tokens), 'expanded_terms': 0}

    for token in query_tokens:
        options = [(token, 1.0)]  # original token
        if should_expand_token(token, stopwords):
            expanded = expand_query_enhanced(token, word_model, topn=5, similarity_threshold=similarity_threshold)
            for exp_token, sim in expanded[1:]:  # skip the original itself
                if exp_token != token and exp_token not in stopwords:
                    weight = sim * expansion_weight
                    options.append((exp_token, weight))
                    expansion_stats['expanded_terms'] += 1
        token_options.append(options)

    # Step 5: Generate and weight n-grams with enhanced scoring
    word_counts = {}
    ngram_details = {}  # Store additional info for debugging

    for n in range(1, ngram_max + 1):
        for i in range(len(token_options) - n + 1):
            options_slice = token_options[i:i + n]
            for gram_tuple in product(*options_slice):
                tokens, weights = zip(*gram_tuple)
                
                
                tokens_clean = [t.replace(" ", "_") for t in tokens]
                gram = " ".join(tokens_clean)

                # Use enhanced weighting
                length_scaling = (n / original_query_length) ** 0.5
                final_weight = length_scaling * sum(weights) / len(weights)

                word_counts[gram] = word_counts.get(gram, 0) + final_weight
                ngram_details[gram] = {
                    'length': n,
                    'tokens': tokens
                }

    # Step 6: Filter redundant n-grams
    word_counts = filter_redundant_ngrams(word_counts)
    
    # Step 7: Keep only top n-grams to reduce noise
    if len(word_counts) > max_ngrams:
        top_ngrams = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_ngrams]
        word_counts = dict(top_ngrams)

    total_weight = sum(word_counts.values())
    if total_weight == 0:
        return []

    # Step 8: Build query vector
    feature_names = vectorizer.get_feature_names_out().tolist()
    col_index = {term.lower(): idx for idx, term in enumerate(feature_names)}
    query_vector = np.zeros(len(feature_names))

    for token, weight in word_counts.items():
        idx = col_index.get(token)
        if idx is not None:
            query_vector[idx] = weight / total_weight  # normalized weight

    # Step 9: Cosine similarity
    cosine_sim = cosine_similarity([query_vector], tfidf_matrix)[0]
    ranked = sorted(zip(article_ids, cosine_sim), key=lambda x: x[1], reverse=True)

    # Enhanced stats for debugging
    enhanced_stats = {
        **expansion_stats,
        'final_ngrams': len(word_counts),
    }

    return ranked, enhanced_stats, word_counts


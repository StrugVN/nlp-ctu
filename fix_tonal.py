import re
from functools import lru_cache
from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordDetokenizer
import kenlm


# Remove Vietnamese accents
def remove_vn_accent(word):
    word = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', word)
    word = re.sub('[éèẻẽẹêếềểễệ]', 'e', word)
    word = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', word)
    word = re.sub('[íìỉĩị]', 'i', word)
    word = re.sub('[úùủũụưứừửữự]', 'u', word)
    word = re.sub('[ýỳỷỹỵ]', 'y', word)
    word = re.sub('đ', 'd', word)
    return word

# Load Vietnamese syllables once into a lookup
def build_accent_lookup(filepath='all-vietnamese-syllables.txt'):
    lookup = defaultdict(set)
    with open(filepath, encoding='utf-8') as f:
        for w in f.read().splitlines():
            key = remove_vn_accent(w.lower())
            lookup[key].add(w.lower())
    return lookup

# Global accent map
ACCENT_LOOKUP = build_accent_lookup()

# Cached variant generator
@lru_cache(maxsize=10000)
def gen_accents_word_cached(word):
    key = remove_vn_accent(word.lower())
    return ACCENT_LOOKUP.get(key, {word})

def beam_search_kenlm(words, model, k=3):
    if not needs_diacritic_restoration(" ".join(words)):
        return []

    variants = [list(gen_accents_word_cached(w)) for w in words]
    sequences = [([], 0.0)]  # (sequence_so_far, log_score)

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

# Check if input text contains any Vietnamese diacritics
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
    current_beam = [(0.0, prefix_words[:])]  # (score, token list)

    for target_len in range(len(prefix_words) + 1, len(prefix_words) + top_k + 1):
        next_beam = []
        for score, tokens in current_beam:
            for word in vocab:
                full_seq = tokens + [word]
                full_str = " ".join(full_seq[-3:])  # 3-gram context
                new_score = score + model.score(full_str, bos=False, eos=False)
                next_beam.append((new_score, full_seq))
        # Select the best completed sequence of required length
        best = nlargest(1, [s for s in next_beam if len(s[1]) == target_len], key=lambda x: x[0])
        if best:
            suggestions.append(best[0])
            current_beam = best  # grow from current best
        else:
            break  # no valid candidates left

    return [(detokenize(tokens), score) for score, tokens in suggestions]



if __name__ == "__main__":
    # Example usage
    detokenize = TreebankWordDetokenizer().detokenize
    sentence = "tai nan giao thong tren duong quoc lo"
    print(needs_diacritic_restoration(sentence))
    model = kenlm.Model("vi_model_6gramVinToken.binary")
    result = beam_search_kenlm(sentence.lower().split(), model)
    print("Best:", detokenize(result[0][0]))
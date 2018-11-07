from __future__ import print_function, division

import re
import pickle
import sys
stdout = sys.stdout

from fuzzywuzzy import fuzz
import gensim
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
from scipy.spatial import distance

reload(sys)
sys.setdefaultencoding('latin')
sys.stdout = stdout

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("../models/word2vec.bin", 
                                                                 unicode_errors="ignore", binary=True)

eng_stopwords = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def tokenize(text, stem=False, stopwords=True):
#     words = word_tokenize(text)
    words = text.lower().split()
    for w in words:
        if stopwords:
            if w in eng_stopwords:
                continue
        if stem:
            yield stemmer.stem(w)
        else:
            yield w


def ltokenize(text, stem=False, stopwords=True):
    return list(tokenize(text, stem=stem, stopwords=stopwords))


def clean_txt(text):
    mapper = [
        (r"what's", ""), (r"\'s", " "), (r"\'ve", " have "), (r"can't", "cannot "), (r"n't", " not "), 
        (r"I'm", "I am"), (r"\bm\b", " am "), (r"\'re", " are "), (r"\'d", " would "), (r"\'ll", " will "),
        (r"60k", " 60000 "), (r"\be g\b", " eg "), (r"\bb g\b", " bg "), (r"\0s", "0"), (r"\b9 11\b", "911"),
        (r"e-mail", "email"), (r"\s{2,}", " "), (r"quikly", "quickly"), (r"\busa\b", " America "), 
        (r"\bUSA\b", " America "), (r"\bu s\b", " America "), (r"\buk\b", " england "), (r"imrovement", "improvement"),
        (r"intially", "initially"), (r"\bdms\b", "direct messages "), (r"demonitization", "demonetization"), 
        (r"actived", "active"), (r"kms", " kilometers "), (r"\bcs\b", " computer science "), (r"\bupvotes\b", " up votes "),
        (r"\biphone\b", " phone "), (r"\0rs ", " rs "), (r"calender", "calendar"), (r"ios", "operating system"), 
        (r"programing", "programming"), (r"bestfriend", "best friend"), (r"III", "3"), (r"the US", "America"),
        (r"Astrology", "astrology"), (r"Method", "method"), (r"Find", "find"), (r"banglore", "Banglore"),
        (r"\bJ K\b", " JK ")
    ]
    for pattern, re_text in mapper:
        text = re.sub(pattern, re_text, text, flags=re.IGNORECASE)
    
    return text


def clean_and_tokenize(data, stem=False, stopwords=True):
    data['q1_cleaned'] = data.question1.apply(clean_txt)
    data['q2_cleaned'] = data.question2.apply(clean_txt)
    data['words1_cleaned'] = data.q1_cleaned.apply(lambda x: ltokenize(x, stem=True, stopwords=True))
    data['words2_cleaned'] = data.q2_cleaned.apply(lambda x: ltokenize(x, stem=True, stopwords=True))
    return data


def is_question_key(key):
    if key not in ('question1', 'question2', 'q1_cleaned', 'q2_cleaned'):
        raise ValueError("key must in 'question2', 'question1', 'q1_cleaned', 'q2_cleaned'")
    return True


def is_word_key(key):
    if key not in ('words1', 'words2', 'words1_cleaned', 'words2_cleaned'):
        raise ValueError("key must in 'words1', 'words2', 'words1_cleaned', 'words2_cleaned'")
    return True


def get_shared_word_score(words1, words2):
    word_len = len(words1) + len(words2)
    if word_len == 0:
        return 0
    shared_words = set(words1) & set(words2)
    return len(shared_words) / word_len


def feature_shared_word_score(row, cleaned=False):
    key1, key2 = 'words1', 'words2'
    if cleaned:
        key1 += '_cleaned'
        key2 += '_cleaned'
    w1, w2 = row[key1], row[key2]
    return get_shared_word_score(w1, w2)


def get_question_len(text):
    return len(text)


def feature_question_len(row, key="question1"):
    is_question_key(key)
    return get_question_len(row[key])


def get_words_len(words):
    return len(words)


def feature_words_len(row, key="words1"):
    is_word_key(key)
    return get_words_len(row[key])


def get_is_title(text):
    return int(text.istitle())


def feature_is_title(row, key='question1'):
    is_question_key(key)
    return get_is_title(row[key])


def get_similarty(words1, words2, fn="cosine"):
    if not hasattr(distance, fn):
        raise ValueError("fn must in cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis")
    mat1 = [word2vec_model[w] for w in words1 if w in word2vec_model]
    mat2 = [word2vec_model[w] for w in words2 if w in word2vec_model]
    if not mat1 or not mat2:
        return 0
    vec1 = np.average(mat1, axis=0)
    vec2 = np.average(mat2, axis=0)
    sim = getattr(distance, fn)(vec1, vec2)
    return sim


def feature_cos_similarty(row, fn="cosine", cleaned=False):
    key1, key2 = 'words1', 'words2'
    if cleaned:
        key1 += '_cleaned'
        key2 += '_cleaned'
    words1, words2 = row[key1], row[key2]
    sim = get_similarty(words1, words2, fn)
    return sim


# qratio, wratio, partial_ratio, partial_token_set_ratio, partial_token_sort_ratio, token_set_ratio, token_sort_ratio
def get_fuzz(q1, q2, fn="QRatio"):
    func = getattr(fuzz, fn, None)
    if func is None:
        raise ValueError("fn should be QRatio, WRatio, partial_ratio, partial_token_set_ratio, partial_token_sort_ratio, token_set_ratio, token_sort_ratio")
    return func(q1, q2)


def feature_fuzz(row, fn="QRatio", cleaned=False):
    key1, key2 = 'question1', 'question2'
    if cleaned:
        key1, key2 = 'q1_cleaned', 'q2_cleaned'
    q1 = row[key1]
    q2 = row[key2]
    return get_fuzz(q1, q2, fn=fn)


def sentences2feature(data, cleaned=False, verbose=False):
    key1, key2, key3, key4 = 'question1', 'question2', 'words1', 'words2'
    if cleaned:
        key1, key2, key3, key4 = 'q1_cleaned', 'q2_cleaned', 'words1_cleaned', 'words2_cleaned'
    question_len1 = data.apply(lambda x: feature_question_len(x, key1), axis=1, raw=False)
    question_len2 = data.apply(lambda x: feature_question_len(x, key2), axis=1, raw=False)
    # qratio, wratio, partial_ratio, partial_token_set_ratio, partial_token_sort_ratio, token_set_ratio, token_sort_ratio
    fuzz_qratio = data.apply(lambda x: feature_fuzz(x, fn="QRatio", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("qratio finished.")
    fuzz_wratio = data.apply(lambda x: feature_fuzz(x, fn="WRatio", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("wratio finished.")
    fuzz_partial_ratio = data.apply(lambda x: feature_fuzz(x, fn="partial_ratio", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("partial_ratio finished.")
    fuzz_partial_token_set_ratio = data.apply(lambda x: feature_fuzz(x, fn="partial_token_set_ratio", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("partial_token_set_ratio finished.")
    fuzz_partial_token_sort_ratio = data.apply(lambda x: feature_fuzz(x, fn="partial_token_sort_ratio", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("partial_token_sort_ratio finished.")
    fuzz_token_set_ratio = data.apply(lambda x: feature_fuzz(x, fn="token_set_ratio", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("token_set_ratio finished.")
    fuzz_token_sort_ratio = data.apply(lambda x: feature_fuzz(x, fn="token_sort_ratio", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("token_sort_ratio finished.")
    return pd.DataFrame({
        'question_len1': question_len1,
        'question_len2': question_len2,
        'fuzz_qratio': fuzz_qratio,
        'fuzz_wratio': fuzz_wratio,
        'fuzz_partial_ratio': fuzz_partial_ratio,
        'fuzz_partial_token_set_ratio': fuzz_partial_token_set_ratio,
        'fuzz_partial_token_sort_ratio': fuzz_partial_token_sort_ratio,
        'fuzz_token_set_ratio': fuzz_token_set_ratio,
        'fuzz_token_sort_ratio': fuzz_token_sort_ratio
    })


def tokens2feature(data, cleaned=False, verbose=False):
    key1, key2, key3, key4 = 'question1', 'question2', 'words1', 'words2'
    if cleaned:
        key1, key2, key3, key4 = 'q1_cleaned', 'q2_cleaned', 'words1_cleaned', 'words2_cleaned'
    shared_word_score = data.apply(lambda x: feature_shared_word_score(x, cleaned=cleaned), axis=1, raw=False)
    words_len1 = data.apply(lambda x: feature_words_len(x, key3), axis=1, raw=False)
    words_len2 = data.apply(lambda x: feature_words_len(x, key4), axis=1, raw=False)
    cos_distance = data.apply(lambda x: feature_cos_similarty(x, fn="cosine", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("cosine finished.")
    cityblock_distance = data.apply(lambda x: feature_cos_similarty(x, fn="cityblock", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("cityblock finished.")
    jaccard_distance = data.apply(lambda x: feature_cos_similarty(x, fn="jaccard", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("jaccard finished.")
    canberra_distance = data.apply(lambda x: feature_cos_similarty(x, fn="canberra", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("canberra finished.")
    minkowski_distance = data.apply(lambda x: feature_cos_similarty(x, fn="minkowski", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("minkowski finished.")
    braycurtis_distance = data.apply(lambda x: feature_cos_similarty(x, fn="braycurtis", cleaned=cleaned), axis=1, raw=False)
    if verbose:
        print("braycurtis finished.")
    return pd.DataFrame({
        'shared_word_score': shared_word_score,
        'word_len1': words_len1,
        'word_len2': words_len2,
        'cos_distance': cos_distance,
        'cityblock_distance': cityblock_distance,
        'jaccard_distance': jaccard_distance,
        'canberra_distance': canberra_distance,
        'minkowski_distance': minkowski_distance,
        'braycurtis_distance': braycurtis_distance
    })
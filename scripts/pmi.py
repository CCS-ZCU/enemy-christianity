from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.probability import FreqDist
import numpy as np
from itertools import combinations
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Load data
grouped = pd.read_pickle('../data/large-data/grouped_df.pkl')

# Split by 'lagt_provenience'
grouped_christian = grouped[grouped['lagt_provenience'] == 'christian']
grouped_pagan = grouped[grouped['lagt_provenience'] == 'pagan']

christian_0_300 = grouped_christian[
    (grouped_christian['not_before'] >= 0) & (grouped_christian['not_after'] <= 300)
]
christian_300_600 = grouped_christian[
    (grouped_christian['not_before'] >= 300) & (grouped_christian['not_after'] <= 600)
]
pagan_0_300 = grouped_pagan[
    (grouped_pagan['not_before'] >= 0) & (grouped_pagan['not_after'] <= 300)
]
pagan_300_600 = grouped_pagan[
    (grouped_pagan['not_before'] >= 300) & (grouped_pagan['not_after'] <= 600)
]

# Find most frequent words (lowercased)
def get_fdist(sentences):
    return FreqDist(word.lower() for sent in sentences for word in sent.split())

fdist1 = get_fdist(christian_0_300['lamma_sentence'])
fdist2 = get_fdist(christian_300_600['lamma_sentence'])
fdist3 = get_fdist(pagan_0_300['lamma_sentence'])
fdist4 = get_fdist(pagan_300_600['lamma_sentence'])

# Words that appear >=10 times in all subcorpora
vocab = set(
    word for word in fdist1 if fdist1[word] >= 10
) & set(
    word for word in fdist2 if fdist2[word] >= 10
) & set(
    word for word in fdist3 if fdist3[word] >= 10
) & set(
    word for word in fdist4 if fdist4[word] >= 10
)

def compute_tfidf(sentences):
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    return [tfidf_matrix, feature_names]

tfidf_christian_0_300 = compute_tfidf(christian_0_300['lamma_sentence'])
tfidf_christian_300_600 = compute_tfidf(christian_300_600['lamma_sentence'])
tfidf_pagan_0_300 = compute_tfidf(pagan_0_300['lamma_sentence'])
tfidf_pagan_300_600 = compute_tfidf(pagan_300_600['lamma_sentence'])

def pmi_pair(i, j, top_words, top_indices, col_sums, tfidf_matrix):
    w1, w2 = top_words[i], top_words[j]
    idx1, idx2 = top_indices[i], top_indices[j]
    tfidf_word1 = col_sums[idx1]
    tfidf_word2 = col_sums[idx2]
    rows_with_word1 = tfidf_matrix[:, idx1].nonzero()[0]
    rows_with_word2 = tfidf_matrix[:, idx2].nonzero()[0]
    rows_both = np.intersect1d(rows_with_word1, rows_with_word2)
    joint_tfidf = tfidf_matrix[rows_both, idx1].sum() + tfidf_matrix[rows_both, idx2].sum()
    if joint_tfidf > 0:
        pmi = np.log(joint_tfidf / (tfidf_word1 * tfidf_word2))
        return (w1, w2, pmi)
    return None

def compute_pmi(tfidf_data, pickle_name, top_n=None, max_workers=24):
    """
    Compute weighted PMI for all pairs of words using concurrent.futures.
    tfidf_data: [tfidf_matrix, feature_names]
    top_n: if set, only use top_n words by column sum; else use all words
    max_workers: number of parallel threads
    pickle_path: if set, saves results to this path
    Returns a list of (word1, word2, pmi).
    """
    tfidf_matrix, feature_names = tfidf_data
    col_sums = np.array(tfidf_matrix.sum(axis=0)).flatten()
    if top_n is not None:
        top_indices = np.argsort(col_sums)[-top_n:]
    else:
        top_indices = np.arange(len(feature_names))
    top_words = feature_names[top_indices]
    pairs = list(combinations(range(len(top_words)), 2))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(pmi_pair, i, j, top_words, top_indices, col_sums, tfidf_matrix)
            for i, j in pairs
        ]
        for f in tqdm(as_completed(futures), total=len(futures)):
            r = f.result()
            if r is not None:
                results.append(r)
    with open(f"../data/large-data/{pickle_name}.pkl", "wb") as f:
        pickle.dump(results, f)


compute_pmi(tfidf_christian_0_300, "pmi_christian_0_300")
compute_pmi(tfidf_christian_300_600, "pmi_christian_300_600")
compute_pmi(tfidf_pagan_0_300, "pmi_pagan_0_300")
compute_pmi(tfidf_pagan_300_600, "pmi_pagan_300_600")
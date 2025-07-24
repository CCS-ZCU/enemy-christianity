"""
Precompute and save nearest neighbors for all words in each FastText subcorpus model.
"""

from gensim.models import FastText
from collections import defaultdict
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

def compute_neighbors(model, word, n_neighbors):
    try:
        neighbors = [w for w, _ in model.wv.most_similar(word, topn=n_neighbors)]
    except Exception as e:
        print(f"Error for word '{word}': {e}")
        neighbors = []
    return word, neighbors

def main():
    # Load FastText models
    christian_0_300_model = FastText.load("../data/large-data/fasttext_christian_0_300.model")
    christian_300_600_model = FastText.load("../data/large-data/fasttext_christian_300_600.model")
    pagan_0_300_model = FastText.load("../data/large-data/fasttext_pagan_0_300.model")
    pagan_300_600_model = FastText.load("../data/large-data/fasttext_pagan_300_600.model")

    model_map = {
        "christian_0_300": christian_0_300_model,
        "christian_300_600": christian_300_600_model,
        "pagan_0_300": pagan_0_300_model,
        "pagan_300_600": pagan_300_600_model,
    }
    N_NEIGHBORS = 30

    neighbors_dict = defaultdict(dict)
    for subcorpus, model in model_map.items():
        print(f"Processing {subcorpus}...")
        words = model.wv.index_to_key
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(compute_neighbors, model, word, N_NEIGHBORS): word for word in words}
            for future in as_completed(futures):
                word, neighbors = future.result()
                neighbors_dict[subcorpus][word] = neighbors

    with open("../data/large-data/word_neighbors.pkl", "wb") as f:
        pickle.dump(dict(neighbors_dict), f)
    print("Done. Neighbors saved to ../data/large-data/word_neighbors.pkl")

if __name__ == "__main__":
    main()
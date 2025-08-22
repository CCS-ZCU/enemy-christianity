import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from transformers.pipelines import pipeline

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

def run_bertopic_on_subcorpus(subcorpus, subcorpus_name):
    from bertopic import BERTopic
    from transformers.pipelines import pipeline

    docs = subcorpus['lamma_sentence']

    category_names = [
        "violence, troublemaking",
        "moral depravity",
        "idolatry, heresy, magic",
        "falseness, hypocrisy, inflated self-esteem",
        "evil/dubious agents",
        "general polemical term"
    ]

    embedding_model = pipeline(
        "feature-extraction",
        model="pranaydeeps/Ancient-Greek-BERT",
        #device=-1  # Use CPU
    )

    #Zero-shot mode
    topic_model = BERTopic(verbose=True, 
                            embedding_model=embedding_model,
                            min_topic_size=15,
                            zeroshot_topic_list=category_names,
                            zeroshot_min_similarity=0,
                            representation_model=KeyBERTInspired(),
                            low_memory=True,
                            )

    topic_model.fit(docs)
    topic_model.save(f"../data/large-data/{subcorpus_name}_bertopic_zero-shot_model", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
    labeled_text_df = topic_model.get_document_info(docs)
    labeled_text_df.to_pickle(f"../data/large-data/{subcorpus_name}_bertopic_zero-shot_labeled.pkl")

run_bertopic_on_subcorpus(christian_0_300, "christian_0_300")
run_bertopic_on_subcorpus(christian_300_600, "christian_300_600")
run_bertopic_on_subcorpus(pagan_0_300, "pagan_0_300")
run_bertopic_on_subcorpus(pagan_300_600, "pagan_300_600")

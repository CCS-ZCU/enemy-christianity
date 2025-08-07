import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from transformers.pipelines import pipeline

grouped = pd.read_pickle('../data/large-data/grouped_df.pkl')

category_names = [
    "violence, troublemaking",
    "moral depravity",
    "idolatry, heresy, magic",
    "falseness, hypocrisy, inflated self-esteem",
    "evil/dubious agents",
    "general polemical term"
]

torch.set_num_threads(16)

embedding_model = pipeline(
    "feature-extraction",
    model="pranaydeeps/Ancient-Greek-BERT",
    #device=-1  # Use CPU
)

from umap import UMAP
umap_model = UMAP(n_neighbors=10,
                  metric='cosine',
                  n_components=2,
                  random_state=42,
                    )


#Zero-shot mode
topic_model = BERTopic(verbose=True, 
                        embedding_model=embedding_model,
                        min_topic_size=15,
                        zeroshot_topic_list=category_names,
                        zeroshot_min_similarity=0,
                        representation_model=KeyBERTInspired(),
                        umap_model=umap_model,
                        )


topics, _  = topic_model.fit_transform(grouped['lamma_sentence'])
topic_model.save("../data/large-data/topic_model")

labeled_text_df = topic_model.get_document_info(grouped['lamma_sentence'])
labeled_text_df.to_pickle("../data/large-data/labeled_text_df.pkl")
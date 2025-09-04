import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

df_plot = pd.read_pickle("../data/large-data/df_plot.pkl")
subcorpora = df_plot['subcorpus'].unique().tolist()

def load_neighbors(subcorpus):
    with open("../data/large-data/word_neighbors.pkl", "rb") as f:
        return pickle.load(f).get(subcorpus, {})

st.title("Analysis of the christian and pagan subcorpora (0-300,300-600)")
st.markdown("## t-SNE 2D Plot for nearest Word Neighbors")

word = st.text_input("Word:", value='ἐχθρός')
subcorpus = st.selectbox("Subcorpus:", options=subcorpora)
n_neighbors = st.slider("Neighbors:", min_value=1, max_value=100, value=10)
show_context = st.checkbox("Show context dots", value=True)

if word and subcorpus:
    sub_df = df_plot[df_plot['subcorpus'] == subcorpus]
    neighbors_dict = load_neighbors(subcorpus)
    neighbors = [word] + [n for n, _ in neighbors_dict.get(word, [])[:n_neighbors]]
    filtered = sub_df[sub_df['word'].isin(neighbors)]

    fig = go.Figure()
    if show_context:
        fig.add_trace(go.Scatter(
            x=sub_df['y'], y=sub_df['x'],
            mode='markers', marker=dict(size=3, color='lightgray'),
            name='context', text=sub_df['word'],
            hoverinfo='text', showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=filtered['y'], y=filtered['x'],
        mode='markers+text', marker=dict(size=8, color='red'),
        name='neighbors', text=filtered['word'],
        hoverinfo='text', textposition='top center'
    ))
    fig.update_layout(
        title=f"t-SNE 2D for '{word}' and neighbors in {subcorpus}",
        xaxis_title="t-SNE dim 1", yaxis_title="t-SNE dim 2",
        width=900, height=700,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Nearest neighbors in {subcorpus}:**")
    for i, (n, sim) in enumerate(neighbors_dict.get(word, [])[:n_neighbors], 1):
        st.write(f"{i}. {n} (similarity: {sim:.3f})")

st.markdown("## Tf-idf term cooccurence values")

sele_subcorpus = st.selectbox(
    "Subcorpus:",
    options=["christian_0_300", "christian_300_600", "pagan_0_300", "pagan_300_600"],
    key="tfidf_corp"
)
with open("../data/large-data/tfidf_cooc.pkl", "rb") as f:
    data = pickle.load(f)

cooc_df = data[sele_subcorpus]
fword = st.text_input("Word to show cooccurrences of:", value="ἐχθρός", key="tfidf_word")
filter_btn = st.button("Filter cooccurrences", key="tfidf_btn")

def strongest_cooccurrences(cooc_df, target_word):
    if target_word not in cooc_df.columns:
        st.write(f"'{target_word}' not found in vocabulary.")
        return None
    scores = cooc_df[target_word].drop(target_word).sort_values(ascending=False)
    return scores

if filter_btn:
    result = strongest_cooccurrences(cooc_df, fword)
    if result is not None:
        st.dataframe(result)
else:
    st.dataframe(cooc_df)

st.markdown('''The numbers in the output of strongest_cooccurrences are the TF-IDF weighted co-occurrence scores between word and other words in a vocabulary.

Each value is the sum of products of TF-IDF weights for the two words across all sentences.
Higher values mean the two words tend to appear together in the same sentences, and both are semantically important (high TF-IDF) in those sentences.
It is not a raw frequency, but a measure of semantic association based on TF-IDF.
- Higher score = stronger semantic co-occurrence with your target word.
- Lower score = weaker or less meaningful co-occurrence.
            ''')


st.markdown("## PMI coocurence values")

sele_subcorpus_pmi = st.selectbox(
    "Subcorpus (PMI):",
    options=["christian_0_300", "christian_300_600", "pagan_0_300", "pagan_300_600"],
    key="pmi_corp"
)
with open("../data/large-data/pmi_all_subcorpora.pkl", "rb") as f:
    pmi_data = pickle.load(f)

pmi_df = pmi_data[sele_subcorpus_pmi]
pmi_word = st.text_input("Word to show PMI cooccurrences of:", value="ἐχθρός", key="pmi_word")
pmi_btn = st.button("Filter cooccurrences", key="pmi_btn")

def strongest_pmi(df, target_word):
    mask = (df["word1"] == target_word) | (df["word2"] == target_word)
    sub = df[mask].copy()
    if sub.empty:
        st.write(f"'{target_word}' not found in PMI pairs.")
        return None
    sub["other"] = sub.apply(lambda row: row["word2"] if row["word1"] == target_word else row["word1"], axis=1)
    sub = sub.sort_values("pmi", ascending=False)
    return sub[["other", "pmi", "count"]]

if pmi_btn:
    result = strongest_pmi(pmi_df, pmi_word)
    if result is not None:
        st.dataframe(result)
else:
    st.dataframe(pmi_df)

st.markdown('''column "count" - How many times both words coocur in a sentence.

PMI (Pointwise Mutual Information) measures how much the actual co-occurrence of two words exceeds what would be expected if they were independent.

High PMI: The word pair co-occurs much more often than expected by chance (strong association).

Low/Negative PMI: The pair co-occurs less often than expected (weak or no association).

Zero PMI: The pair co-occurs exactly as expected by chance.
            ''')

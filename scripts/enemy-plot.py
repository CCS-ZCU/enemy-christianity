import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

df_plot = pd.read_pickle("../data/large-data/df_plot.pkl")
subcorpora = df_plot['subcorpus'].unique().tolist()

def load_neighbors(subcorpus):
    with open("../data/large-data/word_neighbors.pkl", "rb") as f:
        return pickle.load(f).get(subcorpus, {})

st.title("t-SNE 2D Plot for Word Neighbors")

word = st.text_input("Word:", value='ἐχθρός')
subcorpus = st.selectbox("Subcorpus:", options=subcorpora)
n_neighbors = st.slider("Neighbors:", min_value=1, max_value=30, value=10)
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
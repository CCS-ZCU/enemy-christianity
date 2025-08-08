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

st.markdown("## PMI values ")

sele_subcorpus = st.selectbox("Subcorpus:", options=["pmi_christian_0_300.pkl", "pmi_christian_300_600.pkl", "pmi_pagan_0_300.pkl", "pmi_pagan_300_600.pkl"])
with open(f"../data/large-data/{sele_subcorpus}", "rb") as f:
    data = pickle.load(f)
corp = pd.DataFrame(data, columns=["0", "1", "pmi"])
fword = st.text_input("Word to show PMI of:", value="ἐχθρός")
filter_btn = st.button("Filter PMI by word")
if filter_btn:
    filtered_df = corp.loc[corp.apply(lambda r: r.str.contains(fword, case=False).any(), axis=1)]
    st.dataframe(filtered_df)
else:
    st.dataframe(corp)

top = st.slider("Top_N matches:", min_value=1, max_value=100, value=20)

def plot_pmi_barchart(df, selected_word='ἐχθρός', top_n=20):
    mask = (df["0"] == selected_word) | (df["1"] == selected_word)
    sub = df[mask].copy()
    sub["other"] = sub.apply(lambda row: row["1"] if row["0"] == selected_word else row["0"], axis=1)
    sub = sub.sort_values("pmi", ascending=False).head(top_n)
    fig = px.bar(
        sub,
        x="pmi",
        y="other",
        orientation="h",
        labels={"pmi": "PMI", "other": "Other Word"},
        title=f"Top {top_n} PMI for '{selected_word}'"
    )
    fig.update_layout(
        width=500,
        height=max(300, top_n * 30),
        yaxis=dict(autorange="reversed")  # Highest PMI at top
    )
    return fig
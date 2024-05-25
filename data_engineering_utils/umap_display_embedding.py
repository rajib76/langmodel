import os

import altair as alt
import cohere
import pandas as pd
import umap
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

co = cohere.Client(os.environ['cohere_api_key'])


def visualize_vector_space(text_vectors, text_embeds, **kwargs):
    vector_description = list(text_vectors.columns)
    n_neighbors = kwargs["n_neighbors"]
    # UMAP is non-linear dimension reduction algorithm. It learns
    # the manifold structure of the data and finds a low dimensional
    # embedding that preserves the essential topological structure
    # of the maniflod.
    dim_reduce = umap.UMAP(n_neighbors=n_neighbors)
    reduced_embeds = dim_reduce.fit_transform(text_embeds)
    df_vector = text_vectors.copy()
    df_vector['x'] = reduced_embeds[:, 0]
    df_vector['y'] = reduced_embeds[:, 1]

    # Plot the vector space
    chart = alt.Chart(df_vector).mark_circle(size=60).encode(
        x=alt.X('x',
                scale=alt.Scale(zero=False)
                ),
        y=alt.Y('y',
                scale=alt.Scale(zero=False)
                ),
        tooltip=vector_description
    ).properties(
        width=700,
        height=400
    )
    return chart


if __name__ == "__main__":
    sentence = [
        """When was tripura admitted to India?""",
        """Tripura merged with India in 1949 and was designated as a 'Part C State' (union territory)""",
        """Tripura became a full-fledged state of India in 1972""",
        """My name is Tripura, I was admitted to Indian Academy of Sports in 1979""",
        """My name is Tripura, I got admitted to Indian national academy of fine arts in 1980"""
    ]
    sentences = pd.DataFrame({'text': sentence
                              })
    emb = co.embed(texts=list(sentences['text']),
                   model='embed-english-v3.0', input_type='search_document').embeddings

    chart = visualize_vector_space(sentences, emb, n_neighbors=2)
    chart.save('chart.html')

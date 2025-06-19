import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sentence_transformers as st_mod  # alias to avoid clash with streamlit


@st.cache_data
def load_data():
    df500 = pd.read_pickle('df500.pkl')
    with open('passage_embeddings.pkl', 'rb') as f:
        passage_embeddings = pickle.load(f)
    with open('nn_model.pkl', 'rb') as f:
        nn_model = pickle.load(f)
    model = st_mod.SentenceTransformer("multi-qa-mpnet-base-cos-v1")
    return df500, passage_embeddings, nn_model, model

df500, passage_embeddings, nn_model, model = load_data()

st.title("Anime Recommendation System")

anime_name = st.selectbox("Select an anime:", df500['Name'].tolist())

def recommend_knn(anime_name):
    chosen_id = df500[df500['Name'] == anime_name].index[0]
    syn = df500.loc[chosen_id, 'sentence']
    query_embedding = model.encode(syn)
    distances, indices = nn_model.kneighbors([query_embedding])
    # second nearest neighbor index (skip the first, which is the same anime)
    second_best_index = indices[0][1]
    return df500.iloc[second_best_index]['Name']


def recommend(anime_name):
    chosen_id = df500[df500['Name'] == anime_name].index[0]
    syn = df500.loc[chosen_id, 'sentence']
    query_embedding = model.encode(syn)
    
    # Compute similarity using model.similarity (returns similarity scores with all passages)
    similarity = model.similarity(query_embedding, passage_embeddings)[0]
    sorted_indices = np.argsort(similarity)
    
    # Get the second highest similarity index (excluding the anime itself)
    second_best_index = sorted_indices[-2].item()
    most_similar_anime = df500.iloc[second_best_index]['Name']
    return most_similar_anime

if st.button("Recommend"):
    sim_recommendation = recommend(anime_name)
    knn_recommendation = recommend_knn(anime_name)
    
    st.write(f"Most similar anime to '{anime_name}' (SIM method): {sim_recommendation}")
    st.write(f"Most similar anime to '{anime_name}' (KNN method): {knn_recommendation}")

##Need to make it faster? embedding takes too long

##Version two: input number of episodes, genre, etc to get an anime recommendation
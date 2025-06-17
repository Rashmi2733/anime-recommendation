


import pandas as pd
import numpy as np
import sentence_transformers as st

# from sentence_transformers import SentenceTransformer

df = pd.read_csv(r"anime-dataset-2023.csv")

# print(df.columns)


final = df.join(df['Genres'].str.get_dummies(sep=', '))
final['Episodes'] = final['Episodes'].replace('UNKNOWN', np.nan).astype(float)
# print(final.sample(3))

def row_to_sentence(row):
    genres = [genre for genre in ['Action', 'Adventure', 'Avant Garde', 'Award Winning', 'Boys Love',
       'Comedy', 'Drama', 'Erotica', 'Fantasy', 'Girls Love',
       'Gourmet',  'Horror', 'Mystery', 'Romance', 'Sci-Fi',
       'Slice of Life', 'Sports', 'Supernatural', 'Suspense']
              if row[genre]]
    genre_str = ', '.join(genres)
    return f"{row['Name']} ({row['Type']}): {row['Synopsis']}, {row['Score']},\
    Genres: {genre_str}, Episodes: {row['Episodes']}, Status: {row['Status']}, \
    Producers: {row['Producers']}, Licensors: {row['Licensors']}, Studios: {row['Studios']}, \
    Source: {row['Source']}, Duration: {row['Duration']}, Rating: {row['Rating']}."

df500 = final[final['Score']!='UNKNOWN'].sort_values(by='Score', ascending = False).head(500)
# df500.sample(3)

sentences500 = df500.apply(row_to_sentence, axis=1).tolist()
df500['sentence'] = sentences500

model = st.SentenceTransformer("multi-qa-mpnet-base-cos-v1")

passage_embeddings = model.encode(sentences500)


anime_name = input('Please input an anime name:')
    # 'Fullmetal Alchemist'

def recommend(anime_name):
    chosen_id = df500[df500['Name']==anime_name].index[0]
    syn = df500['sentence'][chosen_id]
    query_embedding = model.encode(syn)
    # distances, indices = nn_model.kneighbors([query_embedding]) ##Comapring the query embedding but need to add that in the recommend fucntion as well
    similarity = model.similarity(query_embedding, passage_embeddings)[0]
    sorted_indices = np.argsort(similarity)

    second_best_index = sorted_indices[-2].item()
    # print("Index of 2nd most similar item:", second_best_index)
    most_similar_anime = df500.iloc[second_best_index]['Name']
    print(f"Most similar anime to '{anime_name}' (SIM):", most_similar_anime)
    return most_similar_anime

recommend(anime_name)
##Method two
from sklearn.neighbors import NearestNeighbors

nn_model = NearestNeighbors(n_neighbors=2, metric='cosine').fit(passage_embeddings)

def knn_recommend(anime_name):
    chosen_id = df500[df500['Name']==anime_name].index[0]
    syn = df500['sentence'][chosen_id]
    query_embedding2 = model.encode(syn)
    distances, indices = nn_model.kneighbors([query_embedding2]) ##Comapring the query embedding but need to add that in the recommend fucntion as well
    most_similar_anime = df500.iloc[indices[0]]['Name'].tolist()[1:][0]
    print(f"Most similar anime to '{anime_name}' (KNN):", most_similar_anime)
    return most_similar_anime

knn_recommend(anime_name)


import pickle

with open(r'D:\DS\Project\anime-recommendation\passage_embeddings.pkl', 'wb') as f:
    pickle.dump(passage_embeddings, f)

with open(r'D:\DS\Project\anime-recommendation\nn_model.pkl', 'wb') as f:
    pickle.dump(nn_model, f)

# Also save df500 (the filtered dataframe with sentences)
df500.to_pickle(r'D:\DS\Project\anime-recommendation\df500.pkl')


# import ipywidgets as widgets
# from IPython.display import display
# #Dropdown option instead of input option

# anime_dropdown = widgets.Dropdown(
#     options=df500['Name'].tolist(),
#     description='Anime:',
#     style={'description_width': 'initial'},
#     layout=widgets.Layout(width='50%')
# )

# # Output widget
# output = widgets.Output()

# recommend_button = widgets.Button(description="Recommend")

# # Link button to the action
# def on_button_clicked(b):
#     with output:
#         output.clear_output()  # Clear previous output
#         recommend(anime_dropdown.value)
#         knn_recommend(anime_dropdown.value)
#         # print(f"Most similar anime to '{anime_dropdown.value}': {result}") #redundant

# recommend_button.on_click(on_button_clicked)

# # Display
# display(anime_dropdown, recommend_button, output)
"""
Task 1 
Make a semantic search api which in response retrieves top 20 matching words.
Dataset - https://www.kaggle.com/datasets/kkhandekar/calories-in-food-items-per-100-grams
You can extract the column named Fooditem for the search terms.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.title("Semantic Search API")
food_names = pd.read_csv('calories.csv')
food_names = food_names['FoodItem'].tolist() # extract the FoodItem column and convert to python list
st.write(food_names)

@st.cache_data
def prepare_data():
    vectorizer = TfidfVectorizer(stop_words='english') #ignores common words like the, in, and
    food_vectors = vectorizer.fit_transform(food_names) # convert food names to vectors
    return vectorizer, food_vectors
    
def search_food(query, vectorizer, food_vectors):
    query_vec = vectorizer.transform([query])
    simil = cosine_similarity(query_vec, food_vectors) # calculate cosine similarities
    top_indices = simil[0].argsort()[::-1][:20] # top 20 semantically similar items
    return [food_names[i] for i in top_indices]

def main():
    vectorizer, food_vectors = prepare_data() #function call to prepare data
    st.write("Food items loaded successfully.")

    query = st.text_input("Enter food item to search:", on_change=lambda: st.balloons()) # search bar
    
    if query:
        results = search_food(query, vectorizer, food_vectors) #function call to search food
        st.write("Top 20 matching food items:")
        st.table(results)

if __name__ == "__main__":
    main()
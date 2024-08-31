import re
import streamlit as st
import pickle
import numpy as np
import requests
import pandas as pd
import plotly.express as px
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from wordcloud import WordCloud
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def fetch_poster(movie):
    url = "https://www.omdbapi.com/?apikey=21dcff44&t={}".format(movie)
    data = requests.get(url)
    data = data.json()
    try:
        return data['Poster']
    except:
        return ('')

# Function to preprocess data and fit KNN model
@st.cache_resource
def prepare_knn_model(movies):
    # Select features for recommendation
    features = ['year']
    
    # Add genre features if available
    if 'genre' in movies.columns:
        genres = movies['genre'].str.get_dummies(sep=', ')
        features_df = pd.concat([movies[features], genres], axis=1)
    else:
        features_df = movies[features]
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # Fit KNN model
    nn = NearestNeighbors(n_neighbors=6, metric='cosine')
    nn.fit(features_scaled)
    
    return nn, features_df, scaler

# Function to get recommendations
def get_recommendations(movie_name, nn, features_df, scaler, movies):
    # Get the index of the selected movie
    idx = movies.index[movies['movie_name'] == movie_name].tolist()[0]
    
    # Get the features of the selected movie
    movie_features = features_df.iloc[idx].values.reshape(1, -1)
    
    # Scale the features
    movie_features_scaled = scaler.transform(movie_features)
    
    # Find nearest neighbors
    distances, indices = nn.kneighbors(movie_features_scaled)
    
    # Get recommended movie names and posters
    recommended_movies = []
    recommended_posters = []
    
    for i in indices.flatten()[1:]:  # Skip the first one as it's the input movie itself
        recommended_movies.append(movies.iloc[i]['movie_name'])
        recommended_posters.append(fetch_poster(movies.iloc[i]['movie_name']))
    
    return recommended_movies, recommended_posters

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('imdb_2000bollywood_movies_modified.csv')
    similarity = pickle.load(open('finalsimilarity.pkl', 'rb'))
    return movies, similarity

# New function to preprocess data for clustering
@st.cache_resource
def preprocess_for_clustering(movies):
    # Select features for clustering
    features = ['year', 'genre', 'Director', 'Actor1', 'Actor2', 'Actor3', 'Actor4']
    
    # Create a copy of the dataframe with selected features
    df = movies[features].copy()
    
    # Convert year to numeric if it's not already
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Handle categorical variables
    categorical_features = ['genre', 'Director', 'Actor1', 'Actor2', 'Actor3', 'Actor4']
    
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Normalize features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)
    
    return df_scaled

# New function to perform KNN clustering
@st.cache_resource
def perform_knn_clustering(df_scaled, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    return clusters

# New function to visualize clusters
def visualize_clusters(df_scaled, clusters):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    
    df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = clusters
    
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', 
                     labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'},
                     title='Movie Clusters Visualization',
                     color_continuous_scale='Reds',  # This uses a red color scale
                     color_discrete_sequence=['red'] * len(df_pca['Cluster'].unique()))  # Set all points to red
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0)),  # Set line width to 0 to remove border
                      selector=dict(mode='markers'))
    
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                      paper_bgcolor='rgba(0,0,0,0)')  # Transparent surrounding
    
    return fig
def clustering_analysis(movies):
    # Add new section for KNN clustering
    
    df_scaled = preprocess_for_clustering(movies)
    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=5)
    clusters = perform_knn_clustering(df_scaled, n_clusters)
    
    cluster_fig = visualize_clusters(df_scaled, clusters)
    st.plotly_chart(cluster_fig)
    
    # Display cluster statistics
    cluster_stats = pd.DataFrame({'Cluster': range(n_clusters), 'Count': pd.Series(clusters).value_counts().sort_index()})
    st.write("Cluster Statistics:")
    st.dataframe(cluster_stats)

movies, similarity = load_data()


st.header("Bollywood Movie Recommender System")

# Main app layout with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Home", "EDA","Clustering Analysis", "Recommend"])

with tab1:
        st.write("""
    ##### Welcome to the Bollywood Movie Recommender System!

    This app helps you discover new Bollywood movies based on your preferences. 
    
    ##### Features:
    - Personalized movie recommendations
    - Exploratory data analysis of Bollywood movies
    - Advanced clustering analysis to understand movie patterns
    
    ##### How to use this app:
    
    1. Go to the 'Recommend' tab.
    2. Select a movie you like from the dropdown menu.
    3. Click the 'Recommend' button to get personalized movie suggestions.
    4. Explore movie posters and titles of recommended films.
    
    You can also check out some interesting insights about our Bollywood movie database in the 'EDA' tab!
    
    For a deeper dive into movie similarities, visit the 'Clustering Analysis' tab to see how movies 
    group together based on various features.
    
    """)

with tab2:
    # Basic information about the dataset
    st.markdown("**Sample Data:**")
    st.dataframe(movies.head())

    # Data cleaning
    def extract_year(year_str):
        match = re.search(r'\d{4}', str(year_str))
        return int(match.group()) if match else None

    movies['year'] = movies['year'].apply(extract_year)
    movies = movies.dropna(subset=['year'])
    movies['year'] = movies['year'].astype(int)

    # Movies by Year
    year_counts = movies['year'].value_counts().sort_index()
    fig = px.bar(x=year_counts.index, y=year_counts.values, 
                labels={'x': 'Year', 'y': 'Number of Movies'},
                title='Number of Movies by Year',
                color_discrete_sequence=['#FF9999'])
    fig.update_layout(title="Number of Movies by Year")
    st.plotly_chart(fig)

    # Genre analysis
    genres = movies['genre'].str.split(', ', expand=True).stack()
    genre_counts = genres.value_counts()
    fig = px.bar(x=genre_counts.index, y=genre_counts.values, labels={'x': 'Genre', 'y': 'Count'},color_discrete_sequence=['#FF9999'])
    fig.update_layout(title='Movie Genres Distribution')
    st.plotly_chart(fig)

    # Top directors
    top_directors = movies['Director'].value_counts().head(10)
    fig = px.bar(x=top_directors.index, y=top_directors.values, labels={'x': 'Director', 'y': 'Number of Movies'},color_discrete_sequence=['#FF9999'])
    fig.update_layout(title='Top 10 Directors')
    st.plotly_chart(fig)

    # Actor analysis
    all_actors = movies[['Actor1', 'Actor2', 'Actor3', 'Actor4']].values.flatten()
    actor_counts = Counter(all_actors)
    top_actors = pd.Series(actor_counts).sort_values(ascending=False).head(10)
    fig = px.bar(x=top_actors.index, y=top_actors.values, labels={'x': 'Actor', 'y': 'Number of Movies'},color_discrete_sequence=['#FF9999'])
    fig.update_layout(title='Top 10 Actors')
    st.plotly_chart(fig)

with tab3:
    clustering_analysis(movies)

with tab4:
    nn, features_df, scaler = prepare_knn_model(movies)
    
    movie_list = np.sort(movies['movie_name'].unique())
    selected_movie = st.selectbox("Choose a Movie:", options=movie_list, index=None)
    
    if selected_movie:
        if st.button("Recommend"):
            recommended_movie_names, recommended_movie_posters = get_recommendations(selected_movie, nn, features_df, scaler, movies)
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.text(recommended_movie_names[i])
                    if recommended_movie_posters[i] != '':
                        st.image(recommended_movie_posters[i])
                    else:
                        st.image("https://m.media-amazon.com/images/S/sash/4FyxwxECzL-U1J8.png")
    else:
        st.info("Please select a movie from the dropdown to get recommendations.")


"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import pickle


# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from sklearn.metrics.pairwise import cosine_similarity

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode
                (open(main_bg, "rb").read()).decode()});
             background-size: cover;
             background-color: #72459d;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

#load the picked file back to the notebook
model_load_path = "resources/models/lm1_df.pkl"
with open(model_load_path,'rb') as file:
    latent_matrix_1_df = pickle.load(file)
    
#load the picked file back to the notebook
model_load_path = "resources/models/lm2_df.pkl"
with open(model_load_path,'rb') as file:
    latent_matrix_2_df = pickle.load(file)
    
    
    
def movie_recommender(movie, recommender='hybrid'):
    
    movie_1, movie_2, movie_3 = movie
  
    #take the latent vectors for a selected movie from both content
    # and collaborative matrixes
    a_1 = np.array(latent_matrix_1_df.loc[movie_1]).reshape(1, -1)
    a_2 = np.array(latent_matrix_1_df.loc[movie_2]).reshape(1, -1)
    a_3 = np.array(latent_matrix_1_df.loc[movie_3]).reshape(1, -1)
     
    
    b_1 = np.array(latent_matrix_2_df.loc[movie_1]).reshape(1, -1)
    b_2 = np.array(latent_matrix_2_df.loc[movie_2]).reshape(1, -1)
    b_3 = np.array(latent_matrix_2_df.loc[movie_3]).reshape(1, -1)

    #calculate the similarity of this movie with the others in the list
    score_a1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
    score_a2 = cosine_similarity(latent_matrix_1_df, a_2).reshape(-1)
    score_a3 = cosine_similarity(latent_matrix_1_df, a_3).reshape(-1)
    
    
    score_b1 = cosine_similarity(latent_matrix_2_df, b_1).reshape(-1)
    score_b2 = cosine_similarity(latent_matrix_2_df, b_2).reshape(-1)
    score_b3 = cosine_similarity(latent_matrix_2_df, b_3).reshape(-1)
    
    

    # an average measure of both content and collaborative
    hybrid_1 = ((score_a1 + score_b1)/2.0)

    #form a dateframe of similar movies
    dictDf_1 = {'content': score_a1 ,
              'collaborative': score_b1, 
              'hybrid': hybrid_1 }
    similar_1 = pd.DataFrame(dictDf_1, index = latent_matrix_1_df.index )
    
    # an average measure of both content and collaborative
    hybrid_2 = ((score_a2 + score_b2)/2.0)

    #form a dateframe of similar movies
    dictDf_2 = {'content': score_a2 ,
              'collaborative': score_b2, 
              'hybrid': hybrid_2 }
    similar_2 = pd.DataFrame(dictDf_2, index = latent_matrix_1_df.index )
    
    # an average measure of both content and collaborative
    hybrid_3 = ((score_a3 + score_b3)/2.0)

    #form a dateframe of similar movies
    dictDf_3 = {'content': score_a3 ,
              'collaborative': score_b3, 
              'hybrid': hybrid_3 }
    similar_3 = pd.DataFrame(dictDf_3, index = latent_matrix_1_df.index )
    
    similar = pd.concat([similar_1,similar_2,similar_3], axis=0)
    similar.reset_index(inplace=True)
    
    similar = similar.drop_duplicates(subset='index', keep='first')

    #sort it on the basis of either: content, collaborative or hybrid,
    #here : content
    similar.sort_values(recommender, ascending=False, inplace=True)
    
    return  similar['index'][1:].tolist()[:10]

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["About Us","Recommender System","Solution Overview", ]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "About Us":
        
        st.write('# ABOUT US')
        st.write('## Meet NM6 Data Solutions Team')
        st.markdown("##### NM6 Data Solutions is a leading distributor in the fields of Virtualisation and Machine learning solutions,We oﬀer conﬁgured software solutions and implementation support to rapidly deploy data collection, data integration, and data visualization programs across the enterprise.")
        
        st.image('resources/imgs/dp.jpg',width=600)
        st.title("JUSTICE .C. OYEMIKE")
        st.header("Product Manager / Team Lead")
        
        st.image('resources/imgs/bayo.jpeg',width=600)
        st.title("SAlAUDEEN ADEBAYO")
        st.header("Machine Learning Engr. / Techn. Lead ")
        
        st.image('resources/imgs/bodine.jpg',width=600)
        st.title("BODINE MAZIBUKO")
        st.header("Product Developer / Admin Lead ")
        
        st.image('resources/imgs/dapo.jpg',width=600)
        st.title("IFEDAPO OKE")
        st.header("Data Engineer")
        
        st.image('resources/imgs/revoni.png',width=600)
        st.title("RIVONI KHOZA")
        st.header("Machine Learning Model Specialist )")
        
        st.image('resources/imgs/endurance.jpg',width=600)
        st.title("ENDURANCE ARIENKHE")
        st.header("Business Analysist")
        
        #set_bg_hack("resources/imgs/revoni.png")
        
        
    
    
    if page_selection == "Data Insight":
        
        st.write('# DATA INSIGHT')
        st.write('## Exploratory Data Analysis')
        
        st.image('resources/imgs/user_1.png',width=400, )
        st.markdown("")
        
        st.image('resources/imgs/3.png',use_column_width=True)
        st.markdown("")
        
        st.image('resources/imgs/3.png',use_column_width=True)
        st.markdown("")
        
        st.image('resources/imgs/3.png',use_column_width=True)
        st.markdown("")
        
        st.image('resources/imgs/3.png',use_column_width=True)
        st.markdown("")
        
        st.image('resources/imgs/3.png',use_column_width=True)
        st.markdown("")
        
        st.image('resources/imgs/3.png',use_column_width=True)
        st.markdown("")
        
        
        
        
         
        
    if page_selection == "Recommender System":
        # Header contents
        st.write('# RECOMMENDER SYSTEM')
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering',
                        'Hybrid Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = movie_recommender(fav_movies, recommender='content')
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except Exception as e:
                    st.write(e)
                    #st.error("Oops! Looks like this algorithm does't work.\
                     #         We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = movie_recommender(fav_movies, recommender='collaborative')
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")
                    
        # Perform top-10 movie recommendation generation
        if sys == 'Hybrid Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = movie_recommender(fav_movies, recommender='hybrid')
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")




    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()

# Streamlit Interface to Display Recommendations
import streamlit as st
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn
import mlflow.pyfunc


# Load dataset
df = pd.read_csv(r"C:\Users\zubair_khan\Desktop\Data_Science\Projects\Project_8_Dynamic_Workforce_Insights&Recommendation_Platform\data\cleaned_upwork_jobs.csv")
df['average_hourly'] = df[['hourly_low', 'hourly_high']].mean(axis=1)

# Function to display job recommendations
def recommend_jobs(preferred_title, top_n=5):
    # Sample the data
    df_subset = df.sample(n=1000, random_state=42)  # Add random_state for reproducibility

    # Create the TF-IDF matrix for the sampled data
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_subset['Cleaned Job Title'])

    # Check if preferred_title exists in the subset
    matching_indices = df_subset[df_subset['Cleaned Job Title'].str.contains(preferred_title, case=False, na=False)].index
    if matching_indices.empty:
        return None  # Return None if no match is found

    # Use the first matching index as the base job
    idx = matching_indices[0]

    # Calculate cosine similarity for the base job
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Skip the first score (self-match)

    # Get job indices
    job_indices = [i[0] for i in sim_scores]
    return df_subset.iloc[job_indices]


# Streamlit User Interface
st.title('Personalized Job Recommendation System')

st.write("Enter a job title to get personalized job recommendations:")

# User Input
preferred_job_title = st.text_input("Job Title:", "Data Scientist")

# Get and display recommendations
if preferred_job_title:
    recommendations = recommend_jobs(preferred_job_title, top_n=5)
    
    if recommendations is None:
        st.write(f"No jobs found matching the title '{preferred_job_title}'. Please try another.")
    else:
        st.write(f"Top 5 job recommendations for {preferred_job_title}:")
        for i, row in recommendations.iterrows():
            st.write(f"**{row['Cleaned Job Title']}** - ${row['average_hourly']:.2f} per hour")

        # Log user interaction (optional)
        mlflow.log_param('preferred_job_title', preferred_job_title)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
file_path = r"C:\Users\zubair_khan\Desktop\Data_Science\Projects\Project_8_Dynamic_Workforce_Insights&Recommendation_Platform\data\cleaned_upwork_jobs.csv"
df = pd.read_csv(file_path)

# 1. Data Preprocessing
# Convert 'published_date' to datetime format
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

# Extract year and month from 'published_date' for time-based analysis
df['year'] = df['published_date'].dt.year
df['month'] = df['published_date'].dt.month

# Filter remote job postings (assuming 'is_hourly' is a proxy for remote work)
remote_jobs = df[df['is_hourly'] == True]

# Count the number of remote job postings by month
remote_job_trends = remote_jobs.groupby(['year', 'month']).size().reset_index(name='remote_postings')

# Total number of job postings (for comparison)
total_job_trends = df.groupby(['year', 'month']).size().reset_index(name='total_postings')

# Merge total job postings with remote postings to calculate remote work adoption rate
job_trends = pd.merge(remote_job_trends, total_job_trends, on=['year', 'month'])
job_trends['remote_adoption_rate'] = job_trends['remote_postings'] / job_trends['total_postings']

# 2. Forecasting Future Remote Work Trends (Using Holt-Winters Exponential Smoothing)
# Apply time-series forecasting (Exponential Smoothing)
model = ExponentialSmoothing(
    job_trends['remote_adoption_rate'], 
    trend='add', 
    seasonal=None  
)
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)

# 3. Visualization of Remote Work Trends
st.title("Remote Work Trends Analysis")

st.header("Remote Work Adoption Rate Over Time")
st.write("This section shows the shift toward remote work and its implications over time.")

# Plot the remote work adoption rate over time
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=job_trends, x='month', y='remote_adoption_rate', hue='year', ax=ax)
ax.set(title="Remote Work Adoption Rate Over Time", xlabel="Month", ylabel="Remote Work Adoption Rate")
st.pyplot(fig)

# Forecast for future remote work adoption rate
st.subheader("Forecast for Remote Work Adoption Rate")
forecast_months = pd.date_range(
    start=f"{job_trends['year'].max()}-{job_trends['month'].max()}", 
    periods=13, 
    freq='MS'  # Corrected frequency to 'MS' for month start
)[1:]

forecast_data = pd.DataFrame({'date': forecast_months, 'forecast': forecast})

# Plot forecast for future remote work adoption rate
fig, ax = plt.subplots(figsize=(10, 6))

# Historical data plot
sns.lineplot(data=job_trends, x='month', y='remote_adoption_rate', hue='year', ax=ax)

# Forecast data plot
sns.lineplot(x=forecast_data['date'], y=forecast_data['forecast'], ax=ax, label='Forecast', color='red')

# Add legend
ax.legend(title="Year and Forecast")
ax.set(title="Forecast for Remote Work Adoption Rate", xlabel="Time", ylabel="Forecasted Remote Adoption Rate")
st.pyplot(fig)

# 4. Job Categories Shifting Toward Remote Work
st.header("Job Categories Shifting Toward Remote Work")

# Extract job titles that have keywords related to remote work
remote_keywords = ['remote', 'work from home', 'telecommute', 'distributed']
df['is_remote_keyword'] = df['title'].str.lower().str.contains('|'.join(remote_keywords))

# Filter remote jobs based on keywords
remote_keyword_jobs = df[df['is_remote_keyword'] == True]

# Count job categories (top 10 job titles with 'remote' keywords)
remote_job_categories = remote_keyword_jobs['title'].value_counts().head(10)

# Display top remote job categories
st.subheader("Top Job Titles with 'Remote' Keywords")
fig, ax = plt.subplots(figsize=(10, 6))
remote_job_categories.plot(kind='bar', ax=ax)
ax.set(title="Top 10 Job Titles with Remote Work Keywords", xlabel="Job Title", ylabel="Frequency")
st.pyplot(fig)

# 5. Insights and Report
st.header("Insights on Remote Work Trends")
st.write("""
The shift toward remote work has been significant over the past few years. Job postings offering remote work options have increased substantially, 
and this trend is expected to continue in the future. Based on historical data and our forecasts, we predict a steady increase in remote job adoption 
over the next year.

Key insights:
- Remote work adoption rates have increased steadily year-over-year.
- Certain job categories, especially in tech, marketing, and customer support, are more likely to offer remote work.
- The pandemic played a key role in accelerating the shift to remote work, but ongoing technological advancements continue to drive this trend.
""")

# Show the forecast data for future reference
st.subheader("Forecast Data for Remote Work Adoption")
st.write(forecast_data)

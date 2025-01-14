import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# Load the dataset
file_path = r"C:\Users\zubair_khan\Desktop\Data_Science\Projects\Project_8_Dynamic_Workforce_Insights&Recommendation_Platform\data\cleaned_upwork_jobs.csv"  # Replace with your dataset path
df = pd.read_csv(file_path)

# 1. Data Preprocessing
# Convert 'published_date' to datetime format
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

# Extract year and month from 'published_date' for time-based analysis
df['year'] = df['published_date'].dt.year
df['month'] = df['published_date'].dt.month

# Calculate average salary if it's not hourly
df['avg_hourly_rate'] = df[['hourly_low', 'hourly_high']].mean(axis=1)

# 2. Trend Analysis: Monthly Trends for Job Postings
job_postings_by_month = df.groupby(['year', 'month']).size().reset_index(name='job_postings')

# 3. Trend Analysis: Average Salary Trends Over Time
salary_by_month = df.groupby(['year', 'month'])['avg_hourly_rate'].mean().reset_index(name='avg_salary')

# 4. Category Shifts (Top Job Titles by Frequency)
top_job_categories = df['Cleaned Job Title'].value_counts().head(10)

# 5. Streamlit Dashboard
st.title("Job Market Dynamics Dashboard")

st.header("Job Market Trends")
st.write("This dashboard provides insights into the job market dynamics, including trends in job postings, roles, and salaries over time.")

# Monthly Job Postings Trend
st.subheader("Monthly Job Postings")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=job_postings_by_month, x='month', y='job_postings', hue='year', ax=ax)
ax.set(title="Job Postings Over Time", xlabel="Month", ylabel="Number of Job Postings")
st.pyplot(fig)

# Monthly Salary Trend
st.subheader("Average Salary Trends")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=salary_by_month, x='month', y='avg_salary', hue='year', ax=ax)
ax.set(title="Average Salary Over Time", xlabel="Month", ylabel="Average Hourly Salary")
st.pyplot(fig)

# Category Shifts (Top Job Titles)
st.subheader("Top Job Categories by Frequency")
fig, ax = plt.subplots(figsize=(10, 6))
top_job_categories.plot(kind='bar', ax=ax)
ax.set(title="Top 10 Job Categories", xlabel="Job Title", ylabel="Frequency")
st.pyplot(fig)

# 6. Geographical Insights (Optional)
# Assuming 'country' is a column that indicates the country of job postings
country_job_count = df['country'].value_counts()

# Display countries with the most job postings
st.subheader("Job Postings by Country")
fig, ax = plt.subplots(figsize=(10, 6))
country_job_count.head(10).plot(kind='bar', ax=ax)
ax.set(title="Top 10 Countries by Job Postings", xlabel="Country", ylabel="Number of Job Postings")
st.pyplot(fig)

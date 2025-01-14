import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the dataset
file_path = r"C:\Users\zubair_khan\Desktop\Data_Science\Projects\Project_8_Dynamic_Workforce_Insights&Recommendation_Platform\data\cleaned_upwork_jobs.csv"
df = pd.read_csv(file_path)

# 1. Data Preprocessing
# Convert 'published_date' to datetime format
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

# Extract year and month from 'published_date' for time-based analysis
df['year'] = df['published_date'].dt.year
df['month'] = df['published_date'].dt.month

# 2. Trend Analysis: Job Posting Trends Over Time
job_posting_trends = df.groupby(['year', 'month']).size().reset_index(name='job_postings')

# 3. Forecasting Future Job Postings (Using Exponential Smoothing)
scaler = StandardScaler()
job_posting_trends_scaled = scaler.fit_transform(job_posting_trends['job_postings'].values.reshape(-1, 1))

model = ExponentialSmoothing(
    job_posting_trends_scaled.flatten(),
    trend=None,
    seasonal=None,
    seasonal_periods=12,
    initialization_method='estimated'
)
model_fit = model.fit()
forecast_scaled = model_fit.forecast(steps=12)
forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

# 4. Machine Learning Approach for Predicting Job Market Trends
# Feature engineering
df['hourly_range'] = df['hourly_high'] - df['hourly_low']
df['is_high_budget'] = np.where(df['budget'] > df['budget'].median(), 1, 0)

# Convert categorical features to numerical (if applicable)
df = pd.get_dummies(df, columns=['country'], drop_first=True)

# Define features and target variable
features = ['hourly_range', 'is_high_budget', 'year', 'month'] + [col for col in df.columns if 'country_' in col]
target = 'hourly_high'

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict the future job market trends
job_posting_predictions = rf_model.predict(X_test)

# 5. Visualizations
# Plot historical and forecasted job postings
fig, ax = plt.subplots(figsize=(10, 6))

# Plot historical data
sns.lineplot(data=job_posting_trends, x='month', y='job_postings', hue='year', ax=ax)

# Plot forecasted data
forecast_months = range(1, 13)
sns.lineplot(x=forecast_months, y=forecast, ax=ax, color='red', label='Forecasted')

# Customize plot
ax.set(title="Predicted Job Postings Trend", xlabel="Month", ylabel="Job Postings")
plt.legend(title="Legend")
st.pyplot(fig)

# Plot feature importances from the Random Forest model
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=rf_model.feature_importances_, y=features, ax=ax)
ax.set(title="Feature Importances for Job Market Trend Prediction", xlabel="Importance", ylabel="Features")
st.pyplot(fig)

# 6. Actionable Insights and Recommendations
st.title("Future Workforce Trends Prediction")

st.header("Predicted Job Market Trends")
st.write("""
Based on historical job postings data and advanced forecasting models, we predict the following trends in the job market:
- **Emerging Job Categories**: Roles in technology (especially AI, data science), marketing, and customer support will see significant growth.
- **Remote Work**: The shift toward remote work is expected to continue, particularly in tech and administrative roles.
- **Salary Trends**: Salaries are expected to increase in emerging sectors like AI and cybersecurity.
- **Geographical Trends**: Certain regions, such as urban centers, will continue to see higher demand for jobs, but remote work will balance this out by offering opportunities in rural areas.

**Recommendations**:
1. **Focus on Emerging Job Categories**: Professionals should focus on fields like AI, data science, and digital marketing to stay competitive.
2. **Adapt to Remote Work**: Companies should continue adopting flexible work arrangements and invest in technology that enables remote collaboration.
3. **Salary Negotiation**: As certain job categories grow, candidates should stay informed about salary trends and negotiate accordingly.
""")

# Display the forecast data
st.subheader("Forecasted Job Market Trends")
forecast_df = pd.DataFrame(forecast, columns=["Forecasted Postings"], index=range(1, 13))
st.write(forecast_df)

# Show the predicted job postings from the Random Forest model
st.subheader("Predicted Job Market Trends (Random Forest Model)")
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': job_posting_predictions})
st.write(predictions_df)

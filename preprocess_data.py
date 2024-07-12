import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ============================== #
#          LOAD DATA             #
# ============================== #

#Loading the data.Needs to be switched to production snowflake tables from automated notes.
data = pd.read_csv('corr base_2024-07-02T2019.csv')  

# ============================== #
#       PREPROCESSING DATA       #
# ============================== #
# Preprocess the data.
# Select the best features based on the correlation matrix. Link to the matrix found on the app builder.
features = ['Sentiment', 'Resolve & Educate Section Level Score', 
            'Summarize Section Level Score', 'H2 Production QA Model (Voice) Score']

# Ensure no NaN values
data = data.dropna(subset=features + ['SNPS_GROUP'])

# Encode any categorical data in features
for feature in features:
    if data[feature].dtype == 'object':
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])

# Encode categorical target variable
le_target = LabelEncoder()
data['SNPS_GROUP'] = le_target.fit_transform(data['SNPS_GROUP'])

# Save preprocessed data
data.to_csv('preprocessed_data.csv', index=False)

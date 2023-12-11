#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


# In[5]:


import pandas as pd


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


import seaborn as sns


# In[8]:


from scipy.stats import skew, kurtosis


# In[9]:


df = pd.read_csv("C:/Users/Admin/Desktop/PR/ML/1. DataCleaning/Iris.csv")


# In[10]:


print(df.head())


# In[11]:


print(df.tail())


# In[12]:


print(df.info())


# In[13]:


print(df.describe())


# In[14]:


print("Missing values:\n", df.isnull().sum())
print()


# In[15]:


print("Species count:\n", df['Species'].value_counts())
print()


# In[16]:


sns.pairplot(df, hue='Species', markers=["o", "s", "D"])
plt.show()


# In[17]:


plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[1:5]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='Species', y=feature, data=df)
    plt.title(f'{feature} by Species')
plt.tight_layout()
plt.show()


# In[18]:


plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[1:5]):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


# In[19]:


for feature in df.columns[1:5]:
    print(f'{feature} - Skewness: {skew(df[feature])}, Kurtosis: {kurtosis(df[feature])}')


# In[ ]:




Certainly! Data cleaning is a crucial step in the data preparation process that involves identifying and correcting errors, inconsistencies, and inaccuracies in a dataset. The quality of the data greatly influences the results of any analysis or machine learning model, so it's essential to ensure that the data is accurate, reliable, and ready for further processing.

Here's a more detailed theoretical overview of the key aspects of data cleaning:

### 1. **Data Quality Issues:**
   - **Missing Values:** Data may have missing values, which can affect the quality of analysis. Strategies for handling missing values include removal, imputation, or using advanced techniques based on the nature of the data.
   
   - **Duplicates:** Duplicate records can distort analysis results. Identifying and removing duplicates ensures that each observation is unique.

   - **Inconsistent Data:** Inconsistencies in data formats, units, or representations can lead to errors. Standardizing data formats ensures consistency.

   - **Outliers:** Outliers, or extreme values, can skew statistical analyses. Detecting and handling outliers is crucial for maintaining data integrity.

### 2. **Data Cleaning Techniques:**
   - **Handling Missing Values:**
     - Imputation: Fill missing values using statistical measures like mean, median, or mode.
     - Removal: Eliminate rows or columns with missing values.

   - **Dealing with Duplicates:**
     - Identify and remove duplicate records based on unique identifiers.

   - **Correcting Data Types:**
     - Ensure that data types match the nature of the information. For example, dates should be in datetime format, and numerical values should have appropriate types.

   - **Outliers Detection and Handling:**
     - Use statistical methods like z-scores or interquartile range (IQR) to identify outliers.
     - Decide whether to remove outliers or transform them based on the context.

   - **Text Cleaning:**
     - Standardize text data by converting to lowercase.
     - Remove unnecessary whitespaces, special characters, or symbols.

   - **Handling Inconsistent Data:**
     - Standardize data formats and units to ensure consistency.

### 3. **Tools for Data Cleaning:**
   - **Pandas:** A powerful library for data manipulation and analysis in Python. It provides functions for handling missing data, duplicates, and more.

   - **NumPy:** A library for numerical operations in Python, useful for mathematical operations and handling numerical data.

   - **Scikit-learn:** Offers tools for machine learning, including preprocessing techniques that can be useful for outlier detection and imputation.

   - **Matplotlib and Seaborn:** Useful for data visualization, aiding in the identification of outliers and patterns.

### 4. **Best Practices:**
   - **Explore Data:** Understand the characteristics of the data using descriptive statistics and visualizations before cleaning.

   - **Document Changes:** Keep a record of the changes made during the cleaning process for transparency and reproducibility.

   - **Iterative Process:** Data cleaning is often an iterative process. After an initial cleaning, further issues may be discovered during analysis.

### 5. **Conclusion:**
   - Data cleaning is an essential step in the data science workflow, ensuring that data is reliable and suitable for analysis.

Remember that the specific steps and techniques used for data cleaning may vary based on the characteristics of the dataset and the goals of the analysis. Always tailor the data cleaning process to the specific requirements of your project.
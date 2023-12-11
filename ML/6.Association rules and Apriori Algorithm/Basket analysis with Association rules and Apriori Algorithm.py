#!/usr/bin/env python
# coding: utf-8

# # Association rules and Apriori Algorithm

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from pandas.plotting import parallel_coordinates
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('C:/Users/Admin/Desktop/ML/6/basket_analysis.csv')


# In[3]:


df.drop(df.columns[0],axis=1,inplace=True)


# In[4]:


df.shape


# We have 999 basket for us to compute the recommendation for each item that sold in the store. There are 16 items that sold in the shop.

# In[5]:


df.mean()


# We set the mininum support as 0.06, maximum number that being analysed in the basket is 3. We are doing first pruning and see what we get from the result

# In[6]:


# Compute frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(df,
                            min_support = .006,
                            max_len = 3,
                            use_colnames = True)


# In[7]:


# Compute all association rules for frequent_itemsets
rules = association_rules(frequent_itemsets,
                            metric = 'support',
                            min_threshold=0.1)


# In[8]:


#
filtered_rules = rules[(rules['antecedent support'] > 0.02)&
                        (rules['consequent support'] >0.01) &
                        (rules['confidence'] > 0.2) &
                        (rules['lift'] > 1.0)]


# In[9]:


filtered_rules.sort_values('confidence',ascending=False)


# In[10]:


# Generate scatterplot confidence versus support
sns.scatterplot(x = "support", y = "confidence", data = filtered_rules)
plt.show()


# With scatterplot, we can have quick glimpse, where the boundary should be and what metric should be set to filter out the frequent itemsets. 

# In[11]:


filtered_rules = rules[(rules['antecedent support'] > 0.02)&
                        (rules['consequent support'] >0.01) &
                        (rules['confidence'] > 0.45) &
                        (rules['lift'] > 1.0)]


# In[12]:


# Generate scatterplot confidence versus support

sns.scatterplot(x = "support", y = "confidence", size= 'leverage',data = filtered_rules)
plt.legend(bbox_to_anchor= (1.02, 1), loc='upper left',)
plt.show()


# In[13]:


# add extra another rule where support more than 0.2 for given itemset
filtered_rules = rules[(rules['antecedent support'] > 0.02)&
                        (rules['consequent support'] >0.01) &
                        (rules['confidence'] > 0.45) &
                        (rules['lift'] > 1.0)&
                        (rules['support']>0.195)]


# In[14]:


filtered_rules


# In[15]:


def rules_to_coordinates(rules):
    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent:list(antecedent)[0])
    rules['consequent'] = rules['consequents'].apply(lambda consequent:list(consequent)[0])
    rules['rule'] = rules.index
    return rules[['antecedent','consequent','rule']]


# In[16]:


# Convert rules into coordinates suitable for use in a parallel coordinates plot
coords = rules_to_coordinates(filtered_rules)
# Generate parallel coordinates plot
plt.figure(figsize=(3,6))
parallel_coordinates(coords, 'rule',colormap = 'ocean')
plt.legend([])
plt.show()


# From the plot it seems like the butter can be used as cross-selling with other products, it also acts as something to be offered with antecedents that is low. Thus, the customers are more likely to buy them if the butter are offered with cheaper price if they buy the antecedents that sold less in a store

# The sale transaction or count for each unique item approximately for this sample. We will dive into and see whether there is any difference or correlation between the baskets. Since the dataframe is already tabulated one hot data frame, we will straight away and use the dataset to be analyzed with apriori
# ## Apriori Algorithm
# Little bit background introduction for `Apriori Algorithm`. The algorithm assumes that any subset of a frequent itemset must be frequent. Say in our cases, where {apple, unicorn, yoghurt} is frequent then {apple,yoghurt} is frequent. Whereas {apple,unicorn} is not frequent, then {apple,unicorn,yoghurt} is not frequent.
# 
# __SUPPORT__ =  A simple way to control complexity is to place a constraint that such rules must apply to some minimum percentage of the data <br>
# __CONFIDENCE__ =  The probability that B occurs when A; it is p(B|A), which in association mining.<br>
# __LIFT__ =  the co-occurrence of A and B is the probability that we actually see the two together, compared to the probability that we would see the two together if they were unrelated to (independent of) each other.<br>
# __LEVERAGE__ =  alternative is to look at the difference between these quantities rather than their ratio.<br>
# __CONVICTION__ = measure to ascertain the direction of the rule. Unlike lift, conviction is sensitive to the rule direction.
# 
# Just Support and Confidence as a parameter might be misleading for items that are too common/ popular in the basket. It is more likely that popular items are part of the same basket just because they are popular rather than anything else. 

# # What is Apriori Algorithm?
# Apriori Algorithm is one of the algorithm used for transaction data in Association Rule Learning. It allows us to mine the frequent itemset in order to generate association rule between them.
# Example: list of items purchased by customers, details of website which are frequently visited etc.
# 
# This algorithm was introduced by Agrawal and Srikant in 1994.
# 
# Principles behind Apriori Algorithm
# 
# Subset of frequent itemset are frequent itemset.
# Superset of infrequent itemset are infrequent itemset.
# I know you are wondering this is too technical but don’t worry you will get it once we see how it works!
# 
# Apriori Algorithm has three parts:
# 1. Support
# 2. Confidence
# 3. Lift
# 
# Support( I )=
# ( Number of transactions containing item I ) / ( Total number of transactions )
# 
# Confidence( I1 -> I2 ) =
# ( Number of transactions containing I1 and I2 ) / ( Number of transactions containing I1 )
# 
# Lift( I1 -> I2 ) = ( Confidence( I1 -> I2 ) / ( Support(I2) )
# 
# 
Association Rule Learning is a data mining technique which allows us to get interesting insights of relationship among the items.
# In[ ]:





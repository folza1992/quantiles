
# coding: utf-8

# ## Evaluation of Model Performance With Quantiles
# 
# One thing I had to learn quickly in my new job as a data scientist was to evaluate the performance of models in quantiles instead of with a confusion matrix. I hope this little tutorial helps those who are in the same boat!

# In[16]:

#Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from operator import itemgetter 
import os
from sklearn.metrics import confusion_matrix
os.getcwd()
get_ipython().magic('pylab inline')
import time

# In[17]:

#Start clock to get a runbook time at the end. Useful for optimiation of your code
start_time = time.time()


# In[18]:

# Set Seed to get consistent random data
random.seed(101)

# Test Set Seed. for random.seed(101), this number should equal 863
np.random.randint(1000)


# In[19]:

#randomely generate some 'probabilities'. This will be our model output.
propensity_scores = pd.DataFrame(np.random.rand(10000))


# In[20]:

#randomly generate a binary target variable
actual = pd.DataFrame(np.random.randint(2, size = (1,10000))).transpose()


# In[21]:

# Combine and rename our randomly generated data to get two columns and 10,000 rows
df = pd.concat([propensity_scores,actual],axis = 1)
df.columns = ['pos_prob','target']

dec = df


# In[22]:

#You need the output of your models to look something like this. Pos_prob indicates the model score 
# (probability, log_likelyhood, etc.) for each instance, while target indicates the actual dependent variable
# (which for this case will be binary).
print(dec.head())
print(dec.shape)


# In[23]:

# Change a1 for number of bins you want. Example: quartiles would be a1 = 4. Deciles would be a1 = 10
a1 = 4

#function in pandas to put data in to equal bins
dec['quantile'] = pd.qcut(dec['pos_prob'],a1,labels=False)

#This will give us the ranges of the bins later
dec['quantile_values'] = pd.qcut(dec['pos_prob'],a1)

#Only take the positive dependent variable for this exercise
quantiles = dec[dec['target'] == 1]

#subset the Dataframe to only capture the variables that are needed
quantiles = quantiles[['quantile','target','quantile_values']]

#reset index 
dec1 = pd.DataFrame(quantiles.groupby(['quantile','target','quantile_values']).size().reset_index())

#rename columns for new dataframe
dec1.columns = ['quantile','target','quantile_values','freq']

#calculate accuracy of each quantile
dec1['accuracy'] = (dec1['freq'] / (len(dec)/a1))

# The target column is no longer needed, so throw it out to keep results clean
actuals = dec1['target']
del[[dec1['target']]]


# In[24]:

#sort probabilities in descending order to get a visualization of your results, then create a line plot. 
# This results in a very linear orientation, which is expected with random data.
score_values = pd.DataFrame(dec['pos_prob'].sort_values(ascending = False))
x_ax = pd.Series(list(range(len(dec))))
score_values.plot(x = x_ax,color = 'red', xlim = [(len(dec)/(a1*-1)),len(dec)],yticks = np.arange(0,1,.1)
                 ,xticks = np.arange(0,len(dec),len(dec)/a1)
                 ,figsize = (9,7))


# In[25]:

#Create a table showing quantile values, frequency of hits, and accuracy of each quantile. A good model will have higher accuracy
# in the higher bins with scores and lower accuracies in the lower bins:
#Example:
# quantile   accuracy
#   3           .75          
#   2           .50
#   1           .30
#   0           .10

#The results of this model indicate that the model is completely random, which in this case is good because I randomly genderated
# this data to use as an example! 
dec1.sort_values(by = 'quantile',ascending = False)


# ## Just for fun, let's also make a Confusion Matrix to compare.

# In[26]:

#First, grab the actuals and the propensity scores of the data.
cmdf = pd.concat([actual,propensity_scores], axis = 1)
cmdf.columns = ['actuals','propensity_score']


# In[27]:

# Create a little loop that says if our probability score is greater the .5, then predict yes (1), if not then predict no (0)
def predict_outcome(c):
    # a is the name of the output (or propensity score) of the model
    a = 'propensity_score'
    if c[a] >= .5:
        return 1
    else:
        return 0


# In[28]:

#run the loop to create our predicted_outcome variable
cmdf['predicted_outcome'] = cmdf.apply(predict_outcome, axis=1)


# In[29]:

cmdf.head()


# In[30]:

# Display confusion matrix. This is also indicating That our 'model' is completely random. 
cm = confusion_matrix(cmdf['actuals'], cmdf['predicted_outcome'])
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

cm_mat = np.array(cm)
print(cm_mat)


# In[31]:

#Indicates total run time of notebook
print('Total Runtime =',time.time()-start_time,'seconds')


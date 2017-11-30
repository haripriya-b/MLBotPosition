
# coding: utf-8

# In[59]:


import numpy as np
import scipy
from sklearn.externals import  joblib

import pandas as pd
from hmmlearn import hmm
import math


# In[60]:


def loadData(file_name='',data_type='float'):
    data=np.loadtxt('./data/'+file_name, delimiter=',',dtype=data_type)
    return data


# In[61]:


XRAW=loadData('Label.csv')
Obs=loadData('Observations.csv')


# In[62]:


labels_1quad = XRAW
for i in range(len(XRAW)):
    labels_1quad[i,2] += 0
    labels_1quad[i, 3] += 0


# In[65]:


from sklearn.externals import joblib

model = joblib.load("4-10000_12states.pkl") 


# In[66]:


n_states = 12
train_runs = 6000
test_runs = 4000

runs = 10000
steps = 600
lengths_arr_train = np.array(train_runs * [steps])
lengths_arr_test = np.array(test_runs * [steps])



# In[67]:


Obs_aug_train = Obs[:train_runs,1000-steps:]
Obs_row_train = Obs_aug_train.flatten().reshape(-1,1)

Obs_aug_test = Obs[train_runs:,1000-steps:]
Obs_row_test = Obs_aug_test.flatten().reshape(-1,1)


# In[68]:


Z_train = model.predict(Obs_row_train, lengths = lengths_arr_train)

Z_train = np.reshape(Z_train, (train_runs, steps))


# In[69]:


print(Z_train.shape)


# In[70]:


length_labels = len(labels_1quad)
pair = []
state_pairs = []
labels_map = [{} for i in range(length_labels)]

#labels_map[run,{step:[x,y]},{},{}....]
for row in range(length_labels):
    (labels_map[int(labels_1quad[row,0]-1)]).update({labels_1quad[row,1]-1:[labels_1quad[row,2],labels_1quad[row,3]]})


# In[71]:


states_maps = [[] for i in range(n_states)]
for run in range(train_runs):
    labels_steps = labels_map[run] #dictionary {step:[x,y], step:[,]...)
    for step in range(steps):
        if step in labels_steps:
            states_maps[Z_train[run,step]].append(labels_steps[step])


# In[72]:


states_maps_mean = [[]]*n_states
avgX = 0
avgY = 0
count = 0
for i in range(len(states_maps)):
    for j in range(len(states_maps[i])):
        avgX += states_maps[i][j][0]
        avgY += states_maps[i][j][1]
        count+=1
    avgX = avgX/count
    avgY = avgY/count
    states_maps_mean[i]= [avgX,avgY]
    avgX = 0
    avgY = 0
    count = 0


# In[73]:


Z_test = model.predict(Obs_row_test, lengths = lengths_arr_test)

Z_test = np.reshape(Z_test, (test_runs, steps))


# In[74]:


print(Z_test.shape)

thousandth_states = Z_test[:,steps-1]
#print(Z_test[0])

#print(thousandth_states)
next_states = []
for state in thousandth_states:
    next_states.append(np.argmax(model.transmat_[state]))
    


# In[75]:


import csv

with open('output.csv','a') as f:
    headers = ['id','value']
    writer = csv.writer(f)
    writer.writerow(headers)
    for i in range(len(next_states)):
        X,Y = states_maps_mean[next_states[i]]
        xstr = str(6001+i) + "x"
        ystr = str(6001+i) + "y"
        row = [xstr,X-0.2]
        writer.writerow(row)
        row = [ystr,Y-0.2]
        writer.writerow(row)
        

        


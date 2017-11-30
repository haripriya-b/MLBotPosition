
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
from hmmlearn import hmm
import math
import scipy


# In[21]:


def loadData(file_name='',data_type='float'):
    data=np.loadtxt('./data/'+file_name, delimiter=',',dtype=data_type)
    return data


# In[22]:


xraw=loadData('Label.csv')
Obs=loadData('Observations.csv')

# In[23]:

st = 3000
end = 10000
numofsteps= 600

size = end - st
Obs_aug = Obs[st:end,1000-numofsteps:]
lengths_arr = np.array(size * [numofsteps])
#lengths_arr = np.ones(size)
#lengths_arr = lengths_arr*1000
#lengths_arr.astype(int)
print(lengths_arr[:10])
Obs_row = Obs_aug.flatten().reshape(-1,1)
print(Obs_row.shape)

'''
Obs_row=Obs[0]
for i in range(1,len(Obs)):
    Obs_row = np.vstack((Obs_row,Obs[i]))
'''
#Obs_row.reshape(10000000,1)
#Obs_row.shape


# In[24]:


n_states = 6
clf = hmm.GaussianHMM(n_components=n_states, n_iter=60)
clf.fit(Obs_row, lengths = lengths_arr)

#xraw_row.shapex


# In[25]:


from sklearn.externals import joblib
joblib.dump(clf, "./pickles/Soorya/3_10000_600_6states_60iter.pkl")

#joblib.load("filename.pkl") 


# In[12]:


'''
model = joblib.load("model2.pkl") 
#joblib.dump(clf, "4-10000_49states_p2.pkl", protocol=2)

preds = []
for i in range(6000):
    preds.append(model.predict(Obs[i].reshape(-1,1)))
    if i%100 == 0:
        print(i)
    #print(preds)
#model.transmat_
'''


# In[109]:


'''
predicted_class = np.array(preds)

XR=np.zeros((600000,3))
X2=xraw
print(X2.shape)
n = 7
for run in range(6000):
    for i in range(100):
        newx=math.floor(((X2[i,2]+1.5)*n)/3)
        #print newx

        newy=math.floor(((X2[i,3]+1.5)*n)/3)
        #print (newy*10) +newx
        XR[i,2]=(newy*n) +newx
        if(XR[i,2] >= 49):
            print("greater")
        XR[i,0]=X2[i,0]-1
        XR[i,1]=X2[i,1]-1
        
        
        
        #obsidx=math.trunc( X2[i,1])
        #XR[i,1]=Obs[run,obsidx-1]
'''


# In[112]:


'''
act_to_pred_map = {}

for i in range(50):
    act_to_pred_map[i] = []

for i in range(len(XR)):
    dt = XR[i,:]
    #print(dt[2])
    act_to_pred_map[int(dt[2])].append(predicted_class[int(dt[0]),int(dt[1])])

print(act_to_pred_map)
'''


# In[7]:


'''
print(model.transmat_[33])
print(model.means_.shape)
act = {}
for i in range(50):
    act[i] = 0

for i in range(len(XR)):
    dt = XR[i,:]
    #print(dt[2])
    act[int(dt[2])] = act[int(dt[2])] + 1# .append(predicted_class[int(dt[0]),int(dt[1])])
print(act)
'''


# In[15]:


'''
from sklearn.externals import joblib

model = joblib.load("model2.pkl") 
#print(model.transmat_.shape)

Z = 
'''


# In[ ]:


'''
XRAW=loadData('Label.csv')
labels_1quad = XRAW
for i in range(len(XRAW)):
    labels_1quad[i,2] += 1.5
    labels_1quad[i, 3] += 1.5
    

for row in range(length_labels):
    (labels_map[int(labels_1quad[row,0])]).update({labels_1quad[row,1]:[labels_1quad[row,2],labels_1quad[row,3]]})
'''


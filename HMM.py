# coding: utf-8
# In[24]:
#Load text
import numpy as np
import pandas as pd
from hmmlearn import hmm
import math

# In[4]:
def loadData(file_name='',data_type='float'):
    data=np.loadtxt('./data/'+file_name, delimiter=',',dtype=data_type)
    return data
# In[18]:

XRAW=loadData('Label.csv')
Obs=loadData('Observations.csv')
# In[19]:
#XRAW=loadData('Transformed_LabelsCSV.csv')
# In[20]:

XNEW=XRAW[:100,]

# In[21]:
print XNEW.shape
# In[47]:
n_states=100 #multiple of 3
# x= cellnum,
X=np.zeros((100,2))
for i in range(100):
    newx=math.floor((XNEW[i,2]+1.5)/0.3)
    #newx = math.floor(((XNEW[i, 2]) * 100) / 30)
    #print newx
    newy=math.floor((XNEW[i,3]+1.5)/0.3)
    #newy = math.floor(((XNEW[i, 3]) * 100) / 30)
    #print (newy*10) +newx
    X[i,0]=(newy*10)+newx
    obsidx=math.trunc(XNEW[i,1])
    if i == 0:
        print obsidx
    X[i,1]=Obs[0,obsidx-1]
# In[62]:
Xpred=np.zeros((100,2))
for i in range(100):
    Xpred[i,0]=0
    Xpred[i,1]=Obs[1,i]

# In[49]:
clf = hmm.GaussianHMM(n_components=n_states, n_iter=1)
clf.fit(Obs[0])

# In[63]:
# print Xpred.shape
print X.shape


# In[69]:


ZZ=clf.predict(X)


# In[76]:


print ZZ
print clf.transmat_
print clf.score(Xpred)


# In[68]:


XR=np.zeros((100,2))
X2=XRAW[100:200,]
for i in range(100):
    newx=math.floor((X2[i,2]+1.5)/0.3)
    #print newx
    newy=math.floor((X2[i,3]+1.5)/0.3)
    #print (newy*10) +newx
    XR[i,0]=(newy*10) +newx
    obsidx=math.trunc( X2[i,1])
    if i == 0:
        print obsidx
    XR[i,1]=Obs[0,obsidx-1]

#print XR

# In[71]:
print X


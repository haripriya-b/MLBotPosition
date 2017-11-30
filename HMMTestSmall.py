import numpy as np
import scipy
from sklearn.externals import  joblib

import pandas as pd
from hmmlearn import hmm
import math

def loadData(file_name='',data_type='float'):
    data=np.loadtxt('./data/'+file_name, delimiter=',',dtype=data_type)
    return data

runs = 10000
steps = 600
train_runs = 6000
test_runs = 4000
lengths_arr = np.array(runs * [steps])
n_states = 9

print "Loading Data..."
XRAW=loadData('Label.csv')
labels_1quad = XRAW
for i in range(len(XRAW)):
    labels_1quad[i,2] += 1.5
    labels_1quad[i, 3] += 1.5
joblib.dump(labels_1quad, "./pickles/Soorya/labels_1quad.pkl")
Obs=loadData('Observations.csv')
joblib.dump(XRAW, "./pickles/Soorya/XRAW.pkl")
joblib.dump(Obs, "./pickles/Soorya/Obs.pkl")

print "Loading Pickles..."
clf = joblib.load("./pickles/Soorya/4-10000_49states_p2.pkl")
XRAW = joblib.load("./pickles/Soorya/XRAW.pkl")
labels_1quad = joblib.load("./pickles/Soorya/labels_1quad.pkl")
Obs = joblib.load("./pickles/Soorya/Obs.pkl")
print "Done Loading Data..."

print(lengths_arr.shape)

Obs_aug = Obs[:train_runs,1000-steps:]
Obs_row = Obs_aug.flatten().reshape(-1,1)
print(Obs_row.shape)

'''
print "Training..."
clf = hmm.GaussianHMM(n_components=n_states, n_iter=1)
clf.fit(Obs_row, lengths = lengths_arr)
#joblib.dump(clf, "./pickles/Soorya/hmm_model.pkl")
'''

print "Predicting..."
Z = clf.predict(Obs_row,lengths = lengths_arr)
Z = np.reshape(Z, (train_runs, steps))
print "predict shape ", Z.shape
print "labels 1st quad shape ",labels_1quad.shape

for i in range(4000):
    print Z[i,:1]

length_labels = len(labels_1quad)
pair = []
state_pairs = []
labels_map = [{} for i in range(length_labels)]

#labels_map[run,{step:[x,y]},{},{}....]
for row in range(length_labels):
    (labels_map[int(labels_1quad[row,0])]).update({labels_1quad[row,1]:[labels_1quad[row,2],labels_1quad[row,3]]})


states_maps = [[] for i in range(n_states)]
for run in range(train_runs):
    labels_steps = labels_map[run] #dictionary {step:[x,y], step:[,]...)
    for step in range(steps):
        if step in labels_steps:
             [Z[run,step]].append(labels_steps[step])

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
    states_maps_mean[i]= [avgY,avgY]
    avgX = 0
    avgY = 0
    count = 0
    print i, "----", states_maps_mean[i], "\n"


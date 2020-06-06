#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import matplotlib
import matplotlib.dates as md
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
get_ipython().run_line_magic('matplotlib', 'inline')
import time


# In[11]:


#kmeans method elbow curve
#
df = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2014-2019 WE2.xlsx')
data = df[['BGA']]
n_cluster = range(1, 20)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(n_cluster, scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show();


# In[7]:


#OneClass SVM method start
#
#WORKING
#
df = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2014-2018 WE2.xlsx')
#data = df[['Chl','BGA']]
data = df[['BGA']]
scaler = StandardScaler()
np_scaled = scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
# Train OneClass SVM 
outliers_fraction = 0.05
model = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01)
model.fit(data)

# Test OneClass SVM
df = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2019\WE2 2019.xlsx')
#data = df[['Chl','BGA']]
data = df[['BGA']]
df['anomaly'] = pd.Series(model.predict(data))

fig, ax = plt.subplots(figsize=(10,4))

df = df.sort_values('DateTime')

#a = df.loc[(df['anomaly'] == -1) & (df['BGA']>df['Chl']), ['DateTime', 'BGA']] #anomaly
a = df.loc[(df['anomaly'] == -1), ['DateTime', 'BGA']] #anomaly
#b = df.loc[(df['anomaly'] == -1) & (df['BGA']<df['Chl']), ['DateTime', 'Chl']] #anomaly


j11 = datetime.datetime(2019,7,12,11)
k11 = datetime.datetime(2019,7,13,14) 
j12 = datetime.datetime(2019,7,14,12)
k12 = datetime.datetime(2019,7,19,4) 
j13 = datetime.datetime(2019,7,20,12)
k13 = datetime.datetime(2019,7,21,14)
j14 = datetime.datetime(2019,7,22,12)
k14 = datetime.datetime(2019,9,10,12) 

j2 = datetime.datetime(2019,9,19)
k2 = datetime.datetime(2019,9,21)

plt.axvspan(j11,k11, color = 'yellow', zorder = 1)
plt.axvspan(j12,k12, color = 'yellow', zorder = 1)
plt.axvspan(j13,k13, color = 'yellow', zorder = 1)
plt.axvspan(j14,k14, color = 'yellow', zorder = 1)
plt.axvspan(j2,k2, color = 'yellow', zorder = 1)

ax.plot(df['DateTime'], df['BGA'], color='blue', label = 'Phycocyanin', zorder = 10)
#ax.plot(df['DateTime'], df['Chl'], color='green', label = 'Chlorophyll a', zorder = 10)
ax.scatter(a['DateTime'],a['BGA'], color='red', label = 'Anomaly', zorder = 10)
#ax.scatter(b['DateTime'],b['Chl'], color='red', label='_nolegend_', zorder = 10)
plt.ylabel('RFU', fontsize = 20)
plt.xlabel('Date', fontsize = 20)
plt.legend(fontsize = 18)

for tick in ax.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    tick.set_fontsize(16)
plt.show();
#One Class SVM method end


# In[6]:


#Gaussian distribution method start
#
#WORKING
#
#training
df = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2014-2018 WE2.xlsx')
outliers_fraction = 0.05
envelope =  EllipticEnvelope(contamination = outliers_fraction) 
#X_train = df[['Chl','BGA']]
X_train = df[['BGA']]
envelope.fit(X_train)

#testing
df = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2019\WE2 2019.xlsx')
#X_test = df[['Chl','BGA']]
X_test = df[['BGA']]
df['anomaly'] = envelope.predict(X_test)
fig, ax = plt.subplots(figsize=(10, 4))
#a = df.loc[(df['anomaly'] == -1) & (df['BGA']>df['Chl']), ['DateTime', 'BGA']] #anomaly
a = df.loc[(df['anomaly'] == -1), ['DateTime', 'BGA']] #anomaly
#b = df.loc[(df['anomaly'] == -1) & (df['BGA']<df['Chl']), ['DateTime', 'Chl']] #anomaly

j11 = datetime.datetime(2019,7,12,11)
k11 = datetime.datetime(2019,7,13,14) 
j12 = datetime.datetime(2019,7,14,12)
k12 = datetime.datetime(2019,7,19,4) 
j13 = datetime.datetime(2019,7,20,12)
k13 = datetime.datetime(2019,7,21,14)
j14 = datetime.datetime(2019,7,22,12)
k14 = datetime.datetime(2019,9,10,12) 

j2 = datetime.datetime(2019,9,19)
k2 = datetime.datetime(2019,9,21)

plt.axvspan(j11,k11, color = 'yellow', zorder = 1)
plt.axvspan(j12,k12, color = 'yellow', zorder = 1)
plt.axvspan(j13,k13, color = 'yellow', zorder = 1)
plt.axvspan(j14,k14, color = 'yellow', zorder = 1)
plt.axvspan(j2,k2, color = 'yellow', zorder = 1)

ax.plot(df['DateTime'], df['BGA'], color='blue', label='Phycocyanin', zorder = 10)
#ax.plot(df['DateTime'], df['Chl'], color='green', label = 'Chlorophyll a', zorder = 10)
ax.scatter(a['DateTime'],a['BGA'], color='red', label='Anomaly', zorder = 10)
#ax.scatter(b['DateTime'],b['Chl'], color='red', label='_nolegend_', zorder = 10)
plt.ylabel('RFU', fontsize = 20)
plt.xlabel('Date', fontsize = 20)
plt.legend(fontsize = 18)

for tick in ax.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    tick.set_fontsize(16)
plt.show();


# In[12]:


#Isolation Forest Start
#
#Working
#
df = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2014-2018 WE2.xlsx')
#data = df[['Chl','BGA']]
data = df[['BGA']]
scaler = StandardScaler()
np_scaled = scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
# Train Isolation Forest
outliers_fraction = 0.05
model =  IsolationForest(contamination=outliers_fraction)
model.fit(data)

# Test Isolation Forest
df = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2019\WE2 2019.xlsx')
#data = df[['Chl','BGA']]
data = df[['BGA']]
df['anomaly'] = pd.Series(model.predict(data))

fig, ax = plt.subplots(figsize=(10,4))

df = df.sort_values('DateTime')
#a = df.loc[(df['anomaly'] == -1) & (df['BGA']>df['Chl']), ['DateTime', 'BGA']] #anomaly
a = df.loc[(df['anomaly'] == -1), ['DateTime', 'BGA']] #anomaly
#b = df.loc[(df['anomaly'] == -1) & (df['BGA']<df['Chl']), ['DateTime', 'Chl']] #anomaly

j11 = datetime.datetime(2019,7,12,11)
k11 = datetime.datetime(2019,7,13,14) 
j12 = datetime.datetime(2019,7,14,12)
k12 = datetime.datetime(2019,7,19,4) 
j13 = datetime.datetime(2019,7,20,12)
k13 = datetime.datetime(2019,7,21,14)
j14 = datetime.datetime(2019,7,22,12)
k14 = datetime.datetime(2019,9,10,12) 

j2 = datetime.datetime(2019,9,19)
k2 = datetime.datetime(2019,9,21)

plt.axvspan(j11,k11, color = 'yellow', zorder = 1)
plt.axvspan(j12,k12, color = 'yellow', zorder = 1)
plt.axvspan(j13,k13, color = 'yellow', zorder = 1)
plt.axvspan(j14,k14, color = 'yellow', zorder = 1)
plt.axvspan(j2,k2, color = 'yellow', zorder = 1)

ax.plot(df['DateTime'], df['BGA'], color='blue', label = 'Phycocyanin', zorder=10)
#ax.plot(df['DateTime'], df['Chl'], color='green', label = 'Chlorophyll a', zorder=10)
ax.scatter(a['DateTime'],a['BGA'], color='red', label = 'Anomaly', zorder=10)
#ax.scatter(b['DateTime'],b['Chl'], color='red', label='_nolegend_', zorder=10)
plt.ylabel('RFU', fontsize = 20)
plt.xlabel('Date', fontsize = 20)


for tick in ax.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    tick.set_fontsize(16)

x_coordinates = [datetime.datetime(2019,5,15,16,16), datetime.datetime(2019,10,18,15,46)]
y_coordinates = [3.6, 3.6]

plt.plot(x_coordinates, y_coordinates, c = 'black', label = 'RFU Threshold Value', linestyle = 'dashed')
plt.legend(fontsize = 16, loc = 2)    
plt.show();

#Isolation Forest End


# In[4]:


#k-means simplified
df = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2014-2018 WE2.xlsx')
#data = df[['Chl','BGA']]
data = df[['BGA']]
scaler = StandardScaler()
np_scaled = scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
#n_cluster = range(1, 14)
#kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
#scores = [kmeans[i].score(data) for i in range(len(kmeans))]

#fig, ax = plt.subplots(figsize=(6,4))
#ax.plot(n_cluster, scores)
#plt.xlabel('Number of Clusters')
#plt.xticks([0,2,4,6,8,10,12,14])
#plt.ylabel('Score')
#plt.title('Elbow Curve')
#plt.show();

kmeans = KMeans(n_clusters=4, init = 'k-means++',max_iter=5000,n_init=1).fit(data)
df = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2019\WE2 2019.xlsx')
#data = df[['Chl','BGA']]
data = df[['BGA']]
scaler = StandardScaler()
np_scaled = scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
test = kmeans.predict(data)

def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[test[i]]
        distance.at[i] = np.linalg.norm(Xa-Xb)
    return distance

outliers_fraction = 0.05
# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
distance = getDistanceByPoint(data, kmeans)
number_of_outliers = int(outliers_fraction*len(distance))
threshold = distance.nlargest(number_of_outliers).min()
# anomaly1 contains the anomaly result of the above method Cluster (0:normal, 1:anomaly) 
df['anomaly'] = (distance >= threshold).astype(int)

df = df.sort_values('DateTime')

fig, ax = plt.subplots(figsize=(10,4))

j11 = datetime.datetime(2019,7,12,11)
k11 = datetime.datetime(2019,7,13,14) 
j12 = datetime.datetime(2019,7,14,12)
k12 = datetime.datetime(2019,7,19,4) 
j13 = datetime.datetime(2019,7,20,12)
k13 = datetime.datetime(2019,7,21,14)
j14 = datetime.datetime(2019,7,22,12)
k14 = datetime.datetime(2019,9,10,12) 

j2 = datetime.datetime(2019,9,19)
k2 = datetime.datetime(2019,9,21)

plt.axvspan(j11,k11, color = 'yellow', zorder = 1)
plt.axvspan(j12,k12, color = 'yellow', zorder = 1)
plt.axvspan(j13,k13, color = 'yellow', zorder = 1)
plt.axvspan(j14,k14, color = 'yellow', zorder = 1)
plt.axvspan(j2,k2, color = 'yellow', zorder = 1)

#a = df.loc[(df['anomaly'] == 1) & (df['BGA']>df['Chl']), ['DateTime', 'BGA']] #anomaly
a = df.loc[(df['anomaly'] == 1), ['DateTime', 'BGA']] #anomaly
#b = df.loc[(df['anomaly'] == 1) & (df['BGA']<df['Chl']), ['DateTime', 'Chl']] #anomaly
#ax.scatter(df['Chl'], df['BGA'], c=test)
ax.plot(df['DateTime'], df['BGA'], color='blue', label='Phycocyanin', zorder = 10)
#ax.plot(df['DateTime'], df['Chl'], color='green', label = 'Chlorophyll a', zorder = 10)
ax.scatter(a['DateTime'],a['BGA'], color='red', label='Anomaly', zorder = 10)
#ax.scatter(b['DateTime'],b['Chl'], color='red', label='_nolegend_', zorder = 10)
plt.ylabel('RFU', fontsize = 20)
plt.xlabel('Date', fontsize = 20)
plt.legend(fontsize = 18)

for tick in ax.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    tick.set_fontsize(16)
plt.show();


# In[19]:


#Decision function graphs
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Example settings
n_samples = 28000
outliers_fraction = 0.05
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
    ("K-Means Clustering", KMeans(n_clusters=4, init = 'k-means++',max_iter=5000,n_init=1)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Elliptical Envelope", EllipticEnvelope(contamination=outliers_fraction)),
    
    ("Isolation Forest", IsolationForest(behaviour='new',
                                         contamination=outliers_fraction,
                                         random_state=42))]

# Define datasets
df1 = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2014-2019 WE2.xlsx')
datasets = [
    df1[['Chl','BGA']]]

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-10, 20, 150),
                     np.linspace(-10, 20, 150))

df2 = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2014-2019 WE2.xlsx')

plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.02, right=0.98, bottom=.001, top=.96, wspace=.2,
                    hspace=.5)

plot_num = 1
rng = np.random.RandomState(42)
for i_dataset, X in enumerate(datasets):
    # Add outliers
    
    for name, algorithm in anomaly_algorithms:
        plt.subplot(2, 2, plot_num)
        if i_dataset == 0:
            plt.title(name, size=12)

        # fit the data and tag outliers
        if name == "K-Means Clustering":
            t0 = time.time()
            kmeans = algorithm.fit(df1[['Chl','BGA']])
            data = df2[['Chl','BGA']]
            test = algorithm.predict(data)
            
            def getDistanceByPoint():
                distance = pd.Series()
                for i in range(0,len(data)):
                    Xa = np.array(data.loc[i])
                    Xb = kmeans.cluster_centers_[test[i]]
                    distance.at[i] = np.linalg.norm(Xa-Xb)
                return distance

# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
            distance = getDistanceByPoint()
            number_of_outliers = int(outliers_fraction*len(distance))
            threshold = distance.nlargest(number_of_outliers).min()
# anomaly1 contains the anomaly result of the above method Cluster (1:normal, 0:anomaly) 
            y_pred = (distance < threshold).astype(int)
            t1 = time.time()
        else:
            t0 = time.time()
            train = algorithm.fit(X)
            y_pred = train.predict(df2[['Chl','BGA']])
            t1 = time.time()

        # plot the levels lines and the points
        
            
            
        a = df2.loc[(y_pred == 1), ['Chl','BGA']] #normal
        b = df2.loc[(y_pred != 1), ['Chl','BGA']] #anomaly
        
        plt.scatter(a[['Chl']], a[['BGA']], color = 'cyan', s=10, label = 'Normal', zorder = 2)
        plt.scatter(b[['Chl']], b[['BGA']], color = 'red', s=10, label = 'Anomalous', zorder=1)
        
        
        plt.xlim(-1,20)
        plt.ylim(-1,20)
        plt.ylabel('Phycocyanin RFU', fontsize = 10)
        plt.xlabel('Chlorophyll a RFU', fontsize = 10)
        plt.text(.99, .15, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1
        
        if name != "K-Means Clustering":  
            #Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = train.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', zorder = 3)

        else:
            plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', s=11, label = 'Cluster Centres', zorder = 3)
        
        plt.legend(loc='best', fontsize = 12)
plt.show()


# In[11]:


#Time series outlier detection graphs
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

df1 = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2015-2019 WE13.xlsx')
datasets = [
    df1[['Chl','BGA']]]

df2 = pd.read_excel(r'D:\Dropbox\Thesis\Data from Outside Sources\WLE_EXO_chl-BGA - Tom Johengen NOAA\2014-2019 WE4.xlsx')

n_samples = len(df1[['Chl','BGA']])
outliers_fraction = 0.05
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
    ("K-Means Clustering", KMeans(n_clusters=4, init = 'k-means++',max_iter=5000,n_init=1)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Elliptical Envelope", EllipticEnvelope(contamination=outliers_fraction)),
    
    ("Isolation Forest", IsolationForest(behaviour='new',
                                         contamination=outliers_fraction,
                                         random_state=42))]

plt.figure(figsize=(8, 11))
plt.subplots_adjust(left=0.02, right=0.98, bottom=.001, top=.96, wspace=.2,
                    hspace=.2)

plot_num = 1

for i_dataset, X in enumerate(datasets):
    # Add outliers
    
    for name, algorithm in anomaly_algorithms:
        plt.subplot(4, 1, plot_num)
        if i_dataset == 0:
            plt.title(name, size=12, y=0.85, x=0.5)

        # fit the data and tag outliers
        if name == "K-Means Clustering":
            train = algorithm.fit(df1[['Chl','BGA']])
            data = df2[['Chl','BGA']]
            test = train.predict(data)
            def getDistanceByPoint():
                distance = pd.Series()
                for i in range(0,len(data)):
                    Xa = np.array(data.loc[i])
                    Xb = train.cluster_centers_[test[i]]
                    distance.at[i] = np.linalg.norm(Xa-Xb)
                return distance

# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
            distance = getDistanceByPoint()
            number_of_outliers = int(outliers_fraction*len(distance))
            threshold = distance.nlargest(number_of_outliers).min()
# anomaly1 contains the anomaly result of the above method Cluster (1:normal, 0:anomaly) 
            y_pred = (distance < threshold).astype(int)
        else:
            train = algorithm.fit(X)
            y_pred = train.predict(df2[['Chl','BGA']])
        
        a = df2.loc[(y_pred != 1) & (df2['BGA']>df2['Chl']), ['DateTime', 'BGA']] #anomaly
        b = df2.loc[(y_pred != 1) & (df2['BGA']<df2['Chl']), ['DateTime', 'Chl']] #anomaly

        plt.plot(df2['DateTime'], df2['BGA'], color='blue', label='Phycocyanin')
        plt.plot(df2['DateTime'], df2['Chl'], color='green', label = 'Chlorophyll a')
        plt.scatter(a['DateTime'],a['BGA'], color='red', label='Anomaly')
        plt.scatter(b['DateTime'],b['Chl'], color='red', label='_nolegend_')
        plt.xlabel('Date')
        plt.ylabel('RFU')
        plt.legend()
        plot_num += 1
        
plt.show()


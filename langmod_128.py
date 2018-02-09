#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:32:47 2017

@author: snigdha
"""
import numpy as np
import re
import operator
import math
import matplotlib.pyplot as plt
import timeit
from sklearn.utils import shuffle
import pickle

def tokenizeDoc(cur_doc):
    return re.findall('\\w+',cur_doc)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x) + 0.01
    e_x /= np.sum(e_x)
    return e_x

#Reading the train data
line1 = []  
counterDict={}
num_sen = 0
f = open('train.txt','r')
for line in f:
   
   line = "start" + " " + line + " " + "end"
   line1.append(line.lower())
   num_sen = num_sen + 1
   # converting to lower case and splitting based on white space
   words = line.lower().split()
   for word in words:
       if word not in counterDict:
           counterDict[word] = 1
       else:
           counterDict[word] = counterDict[word] + 1
 
#Reading the validation data
line1v = []  
num_senv = 0
fv = open('val.txt','r')
for linev in fv:
   
   linev = "start" + " " + linev + " " + "end"
   line1v.append(linev.lower())
   num_senv = num_senv + 1
                     
      
#Creating a sorted list of tuples        
sorted_x = sorted(counterDict.items(), key=operator.itemgetter(1),reverse = True)

#Creating dictionary of 8000 words
list_dict = []
for i2 in range(0,7999):   
    list_dict.append(sorted_x[i2][0])
  
list_dict.append('unk')

f.close()

#Sentence encoding for train data
s_encode = []

for i1 in range(0,num_sen):

    words1 = line1[i1].split()
    t = len(words1)
    s = []
    for j1 in range(0,t):
        flag = 0    
        for k1 in range(0,(len(list_dict)-1)):
            if(words1[j1] == list_dict[k1]):
                s.append(k1)
                flag = 1                
                
        if(flag == 0):
            s.append(7999)
    
    s1 = np.array(s)
    s_encode.append(s1)   
 

#Sentence encoding for validation data
s_encodev = []

for i1v in range(0,num_senv):

    words1v = line1v[i1v].split()
    tv = len(words1v)
    sv = []
    for j1v in range(0,tv):
        flagv = 0    
        for k1v in range(0,(len(list_dict)-1)):
            if(words1v[j1v] == list_dict[k1v]):
                sv.append(k1v)
                flagv = 1                
                
        if(flagv == 0):
            sv.append(7999)
    
    s1v = np.array(sv)
    s_encodev.append(s1v) 


#Making of 4-grams for training data

grams4 = []
dict_4gr = {}

for l1 in range(0,num_sen):
    a11 = s_encode[l1]
    t1 = len(a11)
    for u1 in range(0,(t1-3)):
        
        g41 = np.array([a11[u1],a11[u1+1],a11[u1+2],a11[u1+3]])
        grams4.append(g41)
        g41_tup = (a11[u1],a11[u1+1],a11[u1+2],a11[u1+3])
        
        
        if g41_tup not in dict_4gr:
          dict_4gr[g41_tup] = 1
        else:
          dict_4gr[g41_tup] = dict_4gr[g41_tup] + 1
        
grams_44 = np.array(grams4)
grams_4 = shuffle(grams_44,random_state=5)
len4g = len(grams_4)

sort_4g = sorted(dict_4gr.items(), key=operator.itemgetter(1),reverse = True)
sorted_4gr = sort_4g[0:50]


with open('all_4grams.pkl','wb') as fr1:
    pickle.dump(sort_4g,fr1)


##Making of 4-grams for validation data

grams4v = []

for l1v in range(0,num_senv):
    a11v = s_encodev[l1v]
    t1v = len(a11v)
    for u1v in range(0,(t1v-3)):
        
        g41v = np.array([a11v[u1v],a11v[u1v+1],a11v[u1v+2],a11v[u1v+3]])
        grams4v.append(g41v)
    
        
grams_4v = np.array(grams4v)
len4gv = len(grams_4v)


#Computing the range matrix for batchsize of 512 for training set
rang = np.int(math.floor(len4g/512))
rang_ar = np.zeros(rang+2)

for p in range(0,(rang+1)):
    
    rang_ar[p] = np.int(512 * p)
           
rang_ar[rang+1] = len4g   
rang_ar = rang_ar.astype(int)   

#Computing the range matrix for batchsize of 512 for validation set
rangv = np.int(math.floor(len4gv/512))
rang_arv = np.zeros(rangv+2)

for pv in range(0,(rangv+1)):
    
    rang_arv[pv] = np.int(512 * pv)
           
rang_arv[rangv+1] = len4gv   
rang_arv = rang_arv.astype(int) 

#Beginning the neural network and initializations
numhid = 128
num_epochs = 100
emb_dim = 16

C = np.random.normal(0, 0.1,(8000,emb_dim))
w1 = np.random.normal(0, 0.1,((emb_dim*3),numhid))
w2 = np.random.normal(0, 0.1,(numhid,8000))
b1= 0
b2= 0
learn_rate = 0.1
    
avg_cr_train = np.zeros(num_epochs)
avg_cr_valid = np.zeros(num_epochs)

perp= np.zeros(num_epochs)
perpv = np.zeros(num_epochs)

cross_entr = np.zeros(rang+1)
cross_entrv = np.zeros(rangv+1)


start = timeit.default_timer()
for iter2 in range(0,num_epochs):
    
    #Training pass
    for i in range(0,(rang+1)):
            
        #Forward pass
        g = grams_4[rang_ar[i]:rang_ar[i+1]]
        lg = len(g)
        xinr = np.zeros((lg,(emb_dim*3)))
        hot_1 = np.zeros((lg,8000))
        sm = np.zeros((lg,8000))  
        for k in range(0,lg):
        
            v1 = g[k,0]
            v2 = g[k,1]
            v3 = g[k,2]        
            v4 = g[k,3]
                
            temp = np.zeros(8000)
            temp[v4] = 1
            hot_1[k,:] = temp.reshape(1,8000)
        
            cx1 = C[v1,:].reshape(1,emb_dim)
            cx2 = C[v2,:].reshape(1,emb_dim)
            cx3 = C[v3,:].reshape(1,emb_dim)
        
            x_input = np.array([cx1,cx2,cx3])
        
            xinr[k,:] = x_input.reshape(1,(emb_dim*3))
                    
        a1 = np.matmul(xinr,w1) + b1        
                      
        a2 = np.matmul(a1,w2) + b2
                       
        for u3 in range(0,lg):
        
            sm[u3,:] =softmax(a2[u3,:])
        
        E = sm - hot_1
           
        #Back pass
            
        deltaw2 = np.matmul(np.transpose(a1),E)/lg
        deltab2 = (np.sum(E,axis = 0)/lg).reshape(1,8000)
            
        temp1 = np.matmul(w2,np.transpose(E))
        deltaw1 = np.transpose(np.matmul(temp1,xinr))/lg
        deltab1 = (np.sum(np.transpose(temp1),axis =0)/lg).reshape(1,numhid)
                   
        w2 = w2 - learn_rate * deltaw2 
        b2 = b2 - learn_rate * deltab2 
        
        w1 = w1 - learn_rate * deltaw1 
        b1 = b1 - learn_rate * deltab1
        
        deltac1 = np.transpose(np.matmul(w1[0:emb_dim,:],temp1))
        
        deltac2 = np.transpose(np.matmul(w1[emb_dim:(emb_dim*2),:],temp1)) 
        
        deltac3 = np.transpose(np.matmul(w1[(emb_dim*2):(emb_dim*3),:],temp1))
        
        for u2 in range(0,lg):
        
            v11 = g[u2,0]
            v22 = g[u2,1]
            v33 = g[u2,2]        
            
            C[v11,:] = C[v11,:] - (learn_rate * deltac1[u2,:])
            C[v22,:] = C[v22,:] - (learn_rate * deltac2[u2,:])
            C[v33,:] = C[v33,:] - (learn_rate * deltac3[u2,:])
        
        temp2 = np.sum(((-1)*np.multiply(hot_1,np.log(sm))),axis = 1)
        cross_entr[i] = np.sum(temp2) 

    #Validation pass
    for iv in range(0,(rangv+1)):
                    
        gv = grams_4v[rang_arv[iv]:rang_arv[iv+1]]
        lgv = len(gv)
        xinrv = np.zeros((lgv,(emb_dim*3)))
        hot_1v = np.zeros((lgv,8000))
        smv = np.zeros((lgv,8000))  
        for kv in range(0,lgv):
        
            v1v = gv[kv,0]
            v2v = gv[kv,1]
            v3v = gv[kv,2]        
            v4v = gv[kv,3]
                
            tempv = np.zeros(8000)
            tempv[v4v] = 1
            hot_1v[kv,:] = tempv.reshape(1,8000)
            
        
            cx1v = C[v1v,:].reshape(1,emb_dim)
            cx2v = C[v2v,:].reshape(1,emb_dim)
            cx3v = C[v3v,:].reshape(1,emb_dim)
        
            x_inputv = np.array([cx1v,cx2v,cx3v])
        
            xinrv[kv,:] = x_inputv.reshape(1,(emb_dim*3))
        
        #Forward pass
        a1v = np.matmul(xinrv,w1) + b1       
                      
        a2v = np.matmul(a1v,w2) + b2
                   
        for u3v in range(0,lgv):
        
            smv[u3v,:] =softmax(a2v[u3v,:])
                
        temp2v = np.sum(((-1)*np.multiply(hot_1v,np.log(smv))),axis = 1)
        cross_entrv[iv] = np.sum(temp2v)
    
      
    avg_cr_train[iter2] = np.sum(cross_entr)/len4g 
                    
    perp[iter2] = math.exp(avg_cr_train[iter2])
    
    avg_cr_valid[iter2] = np.sum(cross_entrv)/len4gv 
                    
    perpv[iter2] = math.exp(avg_cr_valid[iter2])


stop = timeit.default_timer()  
print("Time taken",stop-start)
numar = np.arange(0,num_epochs,1)

plt.figure(1)                
plt.plot(numar, avg_cr_train, 'r', label='Avg Cross Entropy Training Error')
plt.plot(numar, avg_cr_valid, 'b', label='Avg Cross Entropy Validation Error')
plt.title('Observation of average cross-entropy error of training and validation')
plt.xlabel('Number of epochs')
plt.ylabel('Prediction error')
plt.legend()
plt.show()

plt.figure(2)                
plt.plot(numar, perpv, 'b', label='Perplexity on Validation')
plt.title('Observation of perplexity on validation')
plt.xlabel('Number of epochs')
plt.ylabel('Perplexity')
plt.legend()
plt.show()

with open('piku128.pkl','wb') as fr1:
    pickle.dump([w1,w2,b1,b2,C],fr1)
    
with open('sort50_4g.pkl','wb') as fr2:
    pickle.dump(sorted_4gr,fr2)
















    
    
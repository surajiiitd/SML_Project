
# coding: utf-8

# In[ ]:


# from google.colab import drive
# drive.mount('/content/gdrive')


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
import pickle
import scipy.misc
import dlib


# In[2]:


pwd


# # For emotion (unbalanced data)

# In[ ]:


# df = pd.read_csv('fer2013.csv')


# In[ ]:


labels = df['emotion']
data = df['pixels']
data = np.array(data)
labels = np.array(labels)


# In[ ]:


con_data = []
for i in range(len(data)):
  con_data.append([int(d)/255 for d in data[i].split(' ')])


# In[ ]:


con_data = np.reshape(con_data, (len(con_data), 48, 48, 1))


# In[ ]:


train_data, test_data, train_labels, test_labels = train_test_split(con_data, labels, test_size=0.15, random_state=42)


# In[7]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
from sklearn.model_selection import GridSearchCV


# In[ ]:


def buildCovNet(classes):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(10, activation=tf.nn.relu))
  model.add(Dense(10, activation=tf.nn.relu))
#   model.add(Dropout(0.2))
  model.add(Dense(classes, activation=tf.nn.softmax))
  return model


# In[ ]:


# def buildCovNet():
#   model = Sequential()
#   model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
#   model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
#   model.add(MaxPooling2D(pool_size=(2,2)))
#   model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
#   model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
#   model.add(MaxPooling2D(pool_size=(2,2)))
#   model.add(Conv2D(128, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
#   model.add(Conv2D(128, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
#   model.add(Conv2D(128, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
#   model.add(MaxPooling2D(pool_size=(2,2)))
# #   model.add(Conv2D(256, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
# #   model.add(Conv2D(256, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
# #   model.add(Conv2D(256, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
# #   model.add(MaxPooling2D(pool_size=(2,2)))  
# #   model.add(Conv2D(512, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
# #   model.add(Conv2D(512, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
# #   model.add(Conv2D(512, kernel_size=(3,3), data_format='channels_last', input_shape=(48,48,1)))
# #   model.add(MaxPooling2D(pool_size=(2,2)))  
#   model.add(Flatten())
#   model.add(Dense(4096, activation=tf.nn.relu))
#   model.add(Dense(4096, activation=tf.nn.relu))
#   model.add(Dense(1000, activation=tf.nn.relu))
# #   model.add(Dropout(0.2))
#   model.add(Dense(7, activation=tf.nn.softmax))
#   return model


# In[ ]:


def trainNN(train_data, train_labels, test_data, model):
  model.fit(x=train_data, y=train_labels, epochs=10)
  pred_labels = model.predict_classes(test_data)
  return pred_labels


# In[ ]:


def buildNCompile(classes):
  model = buildCovNet(classes)
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model


# In[ ]:


model = buildNCompile(7)
pred_labels = trainNN(train_data, train_labels, test_data, model)


# In[ ]:


def calcAccuracy(test_labels, pred_labels):
    hit = 0
    for i in range(len(test_labels)):
        if test_labels[i] == pred_labels[i]:
            hit += 1
    return hit/len(test_labels)


# In[ ]:


calcAccuracy(test_labels.tolist(), pred_labels.tolist())


# # Emotion (balanced)

# In[10]:


import dlib
from imutils import face_utils
import scipy.misc
import math


# In[ ]:


with open('train_data_fer_6000perclass.txt', 'rb') as reader:
    train_data = pickle.load(reader)


# In[ ]:


with open('train_label_fer_6000perclass.txt', 'rb') as reader:
    train_labels = pickle.load(reader)


# In[11]:


df = pd.read_csv('fer2013.csv')


# In[12]:


df1 = df['pixels'][:28709]


# In[13]:


t_data = []
for i in df1:
    l = i.split(" ")
    t = [int(j) for j in l]
    t_data.append(t)
t_data = np.array(t_data)


# In[30]:


df


# In[37]:


train_label1 = list(df['emotion'][:28709])


# In[14]:


t_data.shape


# In[15]:


import matplotlib.pyplot as plt


# In[19]:


# img = np.array(train_data[200])
# ims = np.reshape(img, (48,48))
img = np.array(t_data[59])
ims = np.reshape(img, (48,48))


# In[20]:


plt.imshow(ims)


# In[21]:


# from google.colab.patches import cv2_imshow


# In[22]:


def calcMeanDistance(x, y, mx, my):
    D = []
    for a, b in zip(x, y):
        d = math.sqrt((a-mx)**2 + (b-my)**2)
        D.append(d)
    return np.array(D)


# In[23]:


def calcRotatedCoord(x_mc, y_mc):
    R = []
    for x, y in zip(x_mc, y_mc):
        r = (math.atan2(y, x)*360)/(2*math.pi)
        R.append(r)
    return np.array(R)


# In[24]:


pwd


# In[25]:


def extractLandmarks(img):
    detector = dlib.get_frontal_face_detector()
#     detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    imr = np.reshape(img, (48,48))
    ims = scipy.misc.toimage(imr)
    np_img = np.array(ims)
    rect = detector(np_img, 1)    # 2nd parameter => # of image pyramid layers to apply
    r = rect[0]
    shape = predictor(np_img, r)
    x_cord = []
    y_cord = []
    for k in range(68):
        x_cord.append(shape.part(k).x)
        y_cord.append(shape.part(k).y)
#         cv2.circle(np_img, (shape.part(k).x, shape.part(k).y), 1, (0,255,0), thickness=1)
#         resized_image = cv2.resize(np_img, (200, 200))
#         cv2_imshow(resized_image)
    x_cord = np.array(x_cord)
    y_cord = np.array(y_cord)
    x_mean = np.mean(x_cord)
    y_mean = np.mean(y_cord)
    x_mc = x_cord - x_mean
    y_mc = y_cord - y_mean
    D = calcMeanDistance(x_cord, y_cord, x_mean, y_mean)
    R = calcRotatedCoord(x_mc, y_mc)
    return x_cord, y_cord, D, R


# In[27]:


all_lmarks_x = []
all_lmarks_y = []
all_D = []
all_R = []
n = 0
faulty = []
for img in t_data:

    try:
        x_cord, y_cord, D, R = extractLandmarks(np.array(img))
    #     print(x_cord)
    #     print(y_cord)
    #     print(D)
    #     print(R)
    #     break
        all_lmarks_x.append(x_cord)
        all_lmarks_y.append(y_cord)
        all_D.append(D)
        all_R.append(R)
    except:
        print(n)
        faulty.append(n)
    n += 1


# In[38]:


train_data=[]
train_label=[]
for (i,row) in enumerate(t_data):
    if i not in faulty:
        train_data.append(row)
        train_label.append(train_label1[i])
        


# In[138]:


main_train_data=[]


# In[139]:


import copy


# In[140]:


# # LBP features
# for i in range(len(all_R)):
#     main_train_data[i]+=all_R[i]
for i in range(len(all_D)):
    main_train_data.append(list(all_R[i])+list(all_D[i])+list(all_lmarks_y[i])+list(all_lmarks_x[i]))
# main_data_set=[]
# main_data_set
# main_data_set+=all_lmarks_x
# main_data_set+=all_lmarks_y
# main_data_set+=all_D
# main_data_set+=all_R


# In[141]:


np.array((main_train_data)).shape


# In[151]:


get_ipython().system('nvidia-smi')


# In[8]:


# Cross Validation

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import csv
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV


# In[6]:


test_data=[]
X_train, X_test, y_train = np.array(train_data),np.array(test_data),np.array(train_label)


# In[9]:


kf=KFold(n_splits=5)
model=[]
score_set1=[]
for train_index, test_index in kf.split(X_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    valid_train_data, valid_test_data = X_train[train_index], X_train[test_index]
    valid_train_label, valid_test_label = y_train[train_index], y_train[test_index]
    
    clf =LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial')
    clf.fit(np.array(valid_train_data),np.array(valid_train_label))

    b=clf.predict(np.array(valid_test_data))

    score1=clf.score(valid_test_data,valid_test_label)
    print("Accuracy ",score1)
    score_set1.append(score1)
    model.append(clf) 


# In[28]:


score_mean=np.mean(score_set1)
standard_dev=np.std(score_set1)
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set1)
best=model[np.argmax(score_set1)]
best_score6=best.score(np.array(train_data),np.array(train_label))
prob_dist6=best.predict_proba(train_data)
label_pred=best.predict(train_data).tolist()

print("Test Accuracy by best model in cross validation : ",best_score6)


# In[4]:


train_data = pickle.load(open("landmarks_detection.pkl","rb"))
train_label= pickle.load(open("label_of landmarks_feature.pkl","rb"))


# In[146]:


# #Dumping into train data and label.
with open("landmarks_detection.pkl", 'wb') as f:
    pickle.dump(main_train_data, f)
with open("faultyindex.pkl", 'wb') as f:
    pickle.dump(faulty, f)
with open("label_of landmarks_feature.pkl", 'wb') as f:
    pickle.dump(train_label, f)


# In[29]:


# ROC Curve 
import copy
import matplotlib.pyplot as plt
prob_dist=np.transpose(prob_dist6)
for i in range(7):
    tpr,fpr=roc_design(prob_dist[i],train_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 0 to 6")
plt.show()


# In[22]:


def roc_design(prob_dist,testdata,checker):
    aux1=[]
    aux2=[]
    testdata1=copy.deepcopy(testdata)
    for i in range(len(testdata)):
        
        aux1.append(prob_dist[i])
        aux2.append(testdata[i])
    main1=sort_list(aux2, aux1)
#     print("Probability in incresing order : ",main1)
    
    tpr=[]
    fpr=[]
    #aux1 has prob_distribution and main1 has testlabel in sorted order
   
    main2=[]
    j=0
    for j in range(len(prob_dist)):
        main2.append(checker)
    i=0
    #Logic 
    if (checker+1)==7:
        flag=checker-1
    else:
        flag=checker+1
        
    while i <len(prob_dist):
        tpr1=0
        fpr1=0
        j=0
        
        while (j  <= i):
            main2[j]=flag
            j=j+1
#         j=i
#         while j <len(prob_dist):
#             main1[j]=2
#             j=j+1
#         print(main1)
#         m=[]
#         tpr.append(tpr1)
#         fpr.append(fpr1)
        tpr1,fpr1=find_tpr_fpr(copy.deepcopy(main2),copy.deepcopy(main1),checker)
#         e.append(testdata)
#         d.append(main1)
        fpr.append(fpr1)
        tpr.append(tpr1)
        
        i=i+50
    return tpr,fpr


# In[11]:


def find_tpr_fpr(predict,real,checker):
    tp=0
    tn=0
    fp=0
#     print("find_tpr_fpr")
    fn=0
    voc=copy.deepcopy([0,1,2,3,4,5,6])
    v=voc.index(checker)
    del voc[v]
    for i in range(len(predict)):
        if predict[i]==checker and real[i]==checker:
            tp=tp+1
        if (predict[i] in voc ) and real[i]==checker:
            fn=fn+1
        if predict[i]==checker and (real[i] in voc):
            fp=fp+1
        if (predict[i] in voc) and (real[i] in voc):
            tn=tn+1
    tpr2=0
    fpr2=0
#     print("Total :",(tp+fp+tn+fn))
    tpr2=float(tp/float(tp+fn))   
    fpr2=float(fp/float(fp+tn))
    
    return tpr2,fpr2


# In[12]:


#geeks for geeks 
def sort_list(list1, list2): 
  
    zipped_pairs = zip(list2, list1) 
  
    z = [x for _, x in sorted(zipped_pairs)] 
      
    return z 


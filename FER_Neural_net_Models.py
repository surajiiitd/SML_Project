
# coding: utf-8

# In[33]:


import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import pandas as pd
import pickle
import numpy.linalg as linalg
from sklearn.model_selection import train_test_split
from random import shuffle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
import seaborn
from random import shuffle
from numpy.random import choice
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from scipy import ndarray
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
from sklearn import preprocessing
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import torchvision.models as models
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.cm as cm


# In[34]:


import warnings
warnings.filterwarnings('ignore')
import tensorflow.keras as keras
import cv2
import tensorflow as tf


# In[35]:


def labelling(predict,true):
    h=[]
    for i in range(len(predict)):
        if predict[i]==true[i]:
            h.append(1)
        else:
            h.append(0)
    return h


# In[36]:


#Cross validation 
def cross_validation_gaussian(train_new,train_label2,train_size):

    fold=5
    model=[]
    score_set=[]
    if len(train_new)%5==0:
        length=int(len(train_new)/5)
    else:
        length=int(len(train_new)/5)+1
    newlength=length
    counter=0
    for q in range(fold):
        valid_test_data=[]
        valid_test_label=[]
        valid_train_data=[]
        valid_train_label=[]
        if (max(len(train_new),length)==length):
            length=len(train_new)
#         print("Counter : ",counter)
#         print("End : ",length)
        for j in range(counter,length):
            valid_test_data.append(train_new[j])
            valid_test_label.append(train_label2[j])
        counter=counter+newlength
        length=length+newlength
        valid_test_data=valid_test_data
        valid_test_label=valid_test_label
        for j in range(len(train_new)):
            if train_new[j] not in valid_test_data:
                valid_train_data.append(train_new[j])
                valid_train_label.append(train_label2[j])
#         print(q)
#         print("Data : ",valid_train_data)
#         clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial').fit(copy.deepcopy(np.array(valid_train_data)), copy.deepcopy(np.array(valid_train_label)))
        clf = GaussianNB().fit(np.array(valid_train_data), np.array(valid_train_label))
        a=[]
        a=clf.predict(np.array(valid_test_data))
        score=clf.score(np.array(valid_test_data),np.array(valid_test_label))
        score_set.append(score)
        model.append(clf) 
        del clf


    return model,score_set
    
    


# In[5]:


#Cross validation 
def cross_validation_svm(train_new,train_label2,train_size):

    fold=5
    model=[]
    score_set=[]
    if len(train_new)%5==0:
        length=int(len(train_new)/5)
    else:
        length=int(len(train_new)/5)+1
    newlength=length
    counter=0
    for q in range(fold):
        valid_test_data=[]
        valid_test_label=[]
        valid_train_data=[]
        valid_train_label=[]
        if (max(len(train_new),length)==length):
            length=len(train_new)
#         print("Counter : ",counter)
#         print("End : ",length)
        for j in range(counter,length):
            valid_test_data.append(train_new[j])
            valid_test_label.append(train_label2[j])
        counter=counter+newlength
        length=length+newlength
        valid_test_data=valid_test_data
        valid_test_label=valid_test_label
        for j in range(len(train_new)):
            if train_new[j] not in valid_test_data:
                valid_train_data.append(train_new[j])
                valid_train_label.append(train_label2[j])
#         print(q)
#         print("Data : ",valid_train_data)
#         clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial').fit(copy.deepcopy(np.array(valid_train_data)), copy.deepcopy(np.array(valid_train_label)))
#         clf = GaussianNB().fit(np.array(valid_train_data), np.array(valid_train_label))
        clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(np.array(valid_train_data), np.array(valid_train_label))
        a=[]
        a=clf.predict(np.array(valid_test_data))
        score=clf.score(np.array(valid_test_data),np.array(valid_test_label))
        score_set.append(score)
        model.append(clf) 
        del clf


    return model,score_set
    
    


# In[6]:


#Cross validation 
def cross_validation_logistic(train_new,train_label2,train_size):

    fold=5
    model=[]
    score_set=[]
    if len(train_new)%5==0:
        length=int(len(train_new)/5)
    else:
        length=int(len(train_new)/5)+1
    newlength=length
    counter=0
    for q in range(fold):
        valid_test_data=[]
        valid_test_label=[]
        valid_train_data=[]
        valid_train_label=[]
        if (max(len(train_new),length)==length):
            length=len(train_new)
#         print("Counter : ",counter)
#         print("End : ",length)
        for j in range(counter,length):
            valid_test_data.append(train_new[j])
            valid_test_label.append(train_label2[j])
        counter=counter+newlength
        length=length+newlength
        valid_test_data=valid_test_data
        valid_test_label=valid_test_label
        for j in range(len(train_new)):
            if train_new[j] not in valid_test_data:
                valid_train_data.append(train_new[j])
                valid_train_label.append(train_label2[j])
#         print(q)
#         print("Data : ",valid_train_data)
        clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial').fit(np.array(valid_train_data), np.array(valid_train_label))
#         clf = GaussianNB().fit(np.array(valid_train_data), np.array(valid_train_label))
        a=[]
        a=clf.predict(np.array(valid_test_data))
        score=clf.score(np.array(valid_test_data),np.array(valid_test_label))
        score_set.append(score)
        model.append(clf) 
        del clf


    return model,score_set
    
    


# In[7]:


def accuracy(predict,true):
    count=0
    for i in range(len(predict)):
        if predict[i]==true[i]:
            count+=1
    return count/float(len(predict))


# In[8]:


#geeks for geeks 
def sort_list(list1, list2): 
  
    zipped_pairs = zip(list2, list1) 
  
    z = [x for _, x in sorted(zipped_pairs)] 
      
    return z 


# In[9]:


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


# In[10]:


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


pwd


# In[12]:


# os.chdir('SML_Project')


# In[13]:


os.chdir('fer2013')


# In[14]:


df=pd.read_csv("fer2013.csv")


# In[15]:


len(df['emotion'].tolist())


# In[16]:


df.head()


# In[17]:


usage=df["Usage"].tolist()
data=df["pixels"].tolist()
train_label1=df["emotion"].tolist()


# In[18]:


train_data=[]
train_label=[]
test_data=[]
test_label=[]
for i in range(len(data)):
    if usage[i]=="Training":
        g=data[i].split(' ')
        h=[int(i) for i in g]
        train_data.append(h)
        train_label.append(train_label1[i])
    else:
      
        g=data[i].split(' ')
        h=[int(i) for i in g]
        test_data.append(h)
        test_label.append(train_label1[i])


# In[ ]:


count=0
t=[0]*7
for i in range(len(train_data)):
    t[train_label[i]]+=1
print(t)


# In[ ]:


size_local=255


# In[ ]:


# #Augmentation Code from https://gist.github.com/tomahim/9ef72befd43f5c106e592425453cb6ae
# def random_rotation(image_array: ndarray):
#     # pick a random degree of rotation between 25% on the left and 25% on the right
#     random_degree = random.uniform(-25, 25)
#     return sk.transform.rotate(image_array, random_degree)

# def random_noise(image_array: ndarray):
#     # add random noise to the image
#     return sk.util.random_noise(image_array)

# def horizontal_flip(image_array: ndarray):
#     # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
#     return image_array[:, ::-1]

# # dictionary of the transformations we defined earlier
# available_transformations = {
#     'rotate': random_rotation,
#     'noise': random_noise,
#     'horizontal_flip': horizontal_flip
# }


# In[ ]:


# folder_path = 'images/cat'
# num_files_desired = 10

# # find all files paths from the folder
# images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# num_generated_files = 0
# while num_generated_files <= num_files_desired:
#     # random image from the folder
#     image_path = random.choice(images)
#     # read image as an two dimensional array of pixels
#     image_to_transform = sk.io.imread(image_path)
#     # random num of transformation to apply
#     num_transformations_to_apply = random.randint(1, len(available_transformations))

#     num_transformations = 0
#     transformed_image = None
#     while num_transformations <= num_transformations_to_apply:
#         # random transformation to apply for a single image
#         key = random.choice(list(available_transformations))
#         transformed_image = available_transformations[key](image_to_transform)
#         num_transformations += 1

# new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)

# # write image to the disk
# io.imsave(new_file_path, transformed_image)
# num_generated_files += 1


# In[ ]:


datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2)


# In[ ]:


# img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
choice=6
#value of i > for choice=0,1,2,3,4,5,6 [0,16,0,0,0,1,0]
#Augmenting the train_data
for j in range(len(train_data)):
    if train_label[j]==choice:
        x=np.array(train_data[j]).reshape(48,48)
        y=[]
        y.append(x)
        y.append(x)
        y.append(x)
        y=np.swapaxes(np.array(y), 0, 2)
        x=copy.deepcopy(np.array(y))
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory

        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='preview', save_prefix='emotion', save_format='jpeg'):
            i += 1
            if i > 2:
#                 print("something")
                break  # otherwise the generator would loop indefinitely
        os.chdir('preview')
        
        lists=os.listdir()
        for k in lists:
            img=cv2.imread(k,1)
            img=img[:,:,0].ravel().tolist()
            train_data.append(img)
            train_label.append(choice)
            
        for k in lists:
            os.remove(k)
        os.chdir('..')


# In[ ]:


pwd


# In[ ]:


len(train_data)


# In[ ]:


train_data1=copy.deepcopy(train_data[:28709])


# In[ ]:


i=28709
while i<len(train_data):
    if i%5000==0:
        print("Counter : ",i)
    a=np.rot90(np.array(train_data[i]).reshape(48,48))
    b=np.rot90(np.array(a))
    c=np.rot90(np.array(b))
    train_data1.append(c.ravel().tolist())
    i+=1
    


# In[ ]:


train_data=copy.deepcopy(train_data1)


# In[ ]:


def retrieve(train,label):
    count=0
    temp=[]
    for i in range(len(list(set(label)))):
        temp.append([])
    for i in range(len(train_data)):
        temp[train_label[i]].append(train_data[i])
    return temp


# In[ ]:


size=6000
data=retrieve(train_data,train_label)


# In[ ]:


temp=[]
for i in range(len(data)):
    
    shuffled_data=random.sample(data[i], len(data[i]))
    X_train=shuffled_data[:size]
    temp.append(X_train)


# In[ ]:


train_data=[]
for i in temp:
    for j in i:
        train_data.append(j)
        


# In[ ]:


train_label=[]
for j in range(7):
    for i in range(6000):
        train_label.append(j)


# In[ ]:


#Random Shuffling of data
main_data_set=[]
for i in range(len(train_data)):

    temp=[]
    temp.append(train_data[i])
    temp.append(train_label[i])
    main_data_set.append(temp)
X_train=random.sample(main_data_set, len(main_data_set))
train_data=[]
train_label=[]
for i in range(len(X_train)):
    train_data.append(X_train[i][0])
    train_label.append(X_train[i][1])


# In[ ]:


pwd


# In[ ]:





# In[ ]:


#For loading the train data and train label
train_data = pickle.load(open("train_data_fer_6000perclass.txt","rb"))
train_label= pickle.load(open("train_label_fer_6000perclass.txt","rb"))


# In[ ]:


# #Dumping into train data and label.
# with open("train_data_fer_6000perclass.txt", 'wb') as f:
#     pickle.dump(train_data, f)
# with open("train_label_fer_6000perclass.txt", 'wb') as f:
#     pickle.dump(train_label, f)


# In[ ]:


#Visualization of Data 
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


# In[ ]:


# #Visualisation of Data 

# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# pca = PCA(n_components=8)
# train_data1=pca.fit_transform(np.array(train_data))
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(np.array(train_data1))


# In[ ]:


# fashion_scatter(tsne_results, np.array(train_label))


# In[ ]:


# train_data=list(train_data)
# train_label=list(train_label)
# test_data=list(test_data)
# test_label=list(test_label)


# In[ ]:


fold=5
train_size=len(train_data)
valid_size=int(train_size/float(fold))


# In[ ]:


# ###################################

# main_train_data=copy.deepcopy(train_data)
# main_test_data=copy.deepcopy(test_data)


# In[ ]:


# train_data=copy.deepcopy(main_train_data)
# test_data=copy.deepcopy(main_test_data)


# In[ ]:


#LBP feature 
radius=3
n_points=8*radius
METHOD="uniform"
train_data1=[]
for i in range(len(train_data)):
    gray=np.array(train_data[i],dtype = np.uint8).reshape(48,48)
    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    train_data1.append(lbp.ravel().tolist())
test_data1=[]
for i in range(len(test_data)):
    gray=np.array(test_data[i],dtype = np.uint8).reshape(48,48)
   
    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    test_data1.append(lbp.ravel().tolist())



# In[ ]:


# # Grayscaling features
# for i in range(len(train_data)):
#     main_train_data[i]+=train_data[i]
# for i in range(len(test_data1)):
#     main_test_data[i]+=test_data[i]
    


# In[ ]:


#Hog Feature
train_data1=[]
for i in range(len(train_data)):
    gray=np.array(train_data[i]).reshape(48,48)
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
    train_data1.append(fd)
test_data1=[]
for i in range(len(test_data)):
    gray=np.array(test_data[i]).reshape(48,48)
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
    test_data1.append(fd)



# In[ ]:


#PCA 
pca = PCA(n_components=30)
train_data1=pca.fit_transform(np.array(train_data1))
test_data1=pca.transform(np.array(test_data1))


# In[ ]:


main_test_data=test_data1
main_train_data=train_data1


# In[ ]:


#PCA 
pca = PCA(n_components=30)
train_data1=pca.fit_transform(np.array(train_data1))
test_data1=pca.transform(np.array(test_data1))


# In[ ]:


test_data=np.hstack((main_test_data,test_data1))
train_data=np.hstack((main_train_data,train_data1))


# In[ ]:


# train=[]
# for i in range(len(train_data)):
#     x=np.array(train_data[i]).reshape(48,48).tolist()
#     y=np.array(x,dtype=np.uint8)
#     y=cv2.resize(y,(224,224)).tolist()
#     temp=[]
#     temp.append(y)
#     temp.append(y)
#     temp.append(y)
#     y=np.swapaxes(np.array(temp), 0, 2)
#     train.append(y)


# In[ ]:


# test=[]
# for i in range(len(test_data)):
#     x=np.array(test_data[i]).reshape(48,48).tolist()
#     y=np.array(x,dtype=np.uint8)
#     y=cv2.resize(y,(224,224)).tolist()
#     temp=[]
#     temp.append(y)
#     temp.append(y)
#     temp.append(y)
#     y=np.swapaxes(np.array(temp), 0, 2)
#     test.append(y)


# In[ ]:


pwd


# In[ ]:


# os.chdir("try1")


# In[ ]:


#Writing into directory named as data
s="Image@"
g=".jpg"
counter=0
for i in range(len(train_data)):
    counter+=1
    f=os.listdir()
    if str(train_label[i]) not in f:
        os.mkdir(str(train_label[i]))
        img=np.array(train_data[i],dtype=np.uint8).reshape(48,48)
        os.chdir(str(train_label[i]))
#         plt.imshow(np.array(train_data[i]).reshape(48,48))
        cv2.imwrite(s+str(counter)+g,img/np.max(img)*255)
        os.chdir('..')
    else:
        img=np.array(train_data[i],dtype=np.uint8).reshape(48,48)
        os.chdir(str(train_label[i]))
        cv2.imwrite(s+str(counter)+g,img/np.max(img)*255)
        os.chdir('..')
        


# In[ ]:


# os.chdir('../try2')


# In[ ]:


#Writing into directory named as testdata
s="Image#"
g=".jpg"
counter=0
for i in range(len(test_data)):
    counter+=1
    f=os.listdir()
    if str(test_label[i]) not in f:
        os.mkdir(str(test_label[i]))
        img=np.array(test_data[i],dtype=np.uint8).reshape(48,48)
        os.chdir(str(test_label[i]))
#         plt.imshow(np.array(train_data[i]).reshape(48,48))
        cv2.imwrite(s+str(counter)+g,img/np.max(img)*255)
        os.chdir('..')
    else:
        img=np.array(test_data[i],dtype=np.uint8).reshape(48,48)
        os.chdir(str(test_label[i]))
        cv2.imwrite(s+str(counter)+g,img/np.max(img)*255)
        os.chdir('..')
        


# In[ ]:


pwd


# In[2]:


import torch.utils.data as data

from PIL import Image
import os
import os.path

def default_loader(path):
	return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    os.chdir(flist)
    f=os.listdir()
    for i in range(len(f)):
        os.chdir(f[i])
        g=os.listdir()
        for j in range(len(g)):
            imlist.append(((os.getcwd()+str('/')+str(g[j])), int(f[i])))

        os.chdir('..')

    return imlist
class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)		
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return img, target

	def __len__(self):
		return len(self.imlist)


# In[1]:


pwd


# In[ ]:


os.chdir('fer2013')


# In[21]:


transform = transforms.Compose(
    [transforms.Resize(224),transforms.ToTensor()])
trainloader = torch.utils.data.DataLoader(ImageFilelist(root="./data/", flist="./data/",
                                                         transform=transform),batch_size=4,
                                          shuffle=True, num_workers=2)
os.chdir('..')
testloader = torch.utils.data.DataLoader(ImageFilelist(root="./testdata/", flist="./testdata/",
                                                         transform=transform),batch_size=4,
                                          shuffle=True, num_workers=2)


# In[ ]:


# t1_label=[]
# for i in range(len(train_label)):
#     t1_label.append([train_label[i]])
    


# In[ ]:


# t2_label=[]
# for i in range(len(test_label)):
#     t2_label.append([test_label[i]])
    


# In[ ]:


pwd


# In[ ]:


pwd


# In[22]:


os.chdir('..')


# In[ ]:


# # Featurs from alexnet
# my_x = np.array(train)
# my_y = np.array(t1_label) # another list of numpy arrays (targets)

# tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
# tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

# trainset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                          shuffle=False, num_workers=2)
# my_x = np.array(test)
# my_y = np.array(t2_label) # another list of numpy arrays (targets)

# tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
# tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

# testset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)


# In[23]:


classes = (0,1,2,3,4,5,6)


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


alexnet = models.alexnet(pretrained=True)
alexnet.classifier=nn.Sequential(*list(alexnet.classifier.children())[:-5])


# In[24]:


vgg16 = models.vgg16(pretrained=True)
vgg16.classifier=nn.Sequential(*list(vgg16.classifier.children())[:-3])


# In[25]:


# alexnet.to("cuda:0")
vgg16.to("cuda:0")


# In[26]:


#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
train_data=[]
train_label=[]
for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs,labels=data
    inputs,labels=inputs.to("cuda:0"), labels.to("cuda:0")

    outputs = vgg16(inputs)
    g=outputs.cpu().detach().numpy().tolist()
    h=labels.cpu().numpy().tolist()
    for j in range(len(g)):

        train_data.append(g[j])
        train_label.append(h[j])
#         print(h[i])
   


# In[ ]:


# #https://pytorch.org/docs/stable/torchvision/models.html
# dataiter = iter(testloader)
# images, labels = dataiter.next()
# images,labels=images.to("cuda:0"), labels.to("cuda:0")
# outputs = alexnet(images)


# In[ ]:


#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
train_data1=[]
for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs,labels=data
    inputs,labels=inputs.to("cuda:0"), labels.to("cuda:0")

    outputs = alexnet(inputs)
    g=outputs.cpu().detach().numpy().tolist()
    h=labels.cpu().numpy().tolist()
    for j in range(len(g)):

        train_data1.append(g[j])
    
#         print(h[i])
   


# In[27]:


test_data=[]
test_label=[]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images,labels=images.to("cuda:0"), labels.to("cuda:0")
        outputs = vgg16(images)
        g=outputs.cpu().numpy().tolist()
        h=labels.cpu().numpy().tolist()
        for i in range(len(g)):
            test_data.append(g[i])
            test_label.append(h[i])


# In[ ]:


test_data1=[]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images,labels=images.to("cuda:0"), labels.to("cuda:0")
        outputs = vgg16(images)
        g=outputs.cpu().numpy().tolist()
        h=labels.cpu().numpy().tolist()
        for i in range(len(g)):
            test_data1.append(g[i])
         


# In[ ]:


test_data=np.hstack((test_data1,test_data))
train_data=np.hstack((train_data1,train_data))


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
train_data1=clf.fit_transform(np.array(train_data),np.array(train_label))
test_data1=clf.transform(np.array(test_data))


# In[28]:


pca = PCA(n_components=1000)
train_data1=pca.fit_transform(np.array(train_data))
test_data1=pca.transform(np.array(test_data))


# In[ ]:


train_data=train_data1
test_data=test_data1


# In[ ]:


train_size=len(train_data)


# In[29]:


#Random Shuffling of data
main_data_set=[]
for i in range(len(train_data)):

    temp=[]
    temp.append(train_data[i])
    temp.append(train_label[i])
    main_data_set.append(temp)
X_train=random.sample(main_data_set, len(main_data_set))
train_data=[]
train_label=[]
for i in range(len(X_train)):
    train_data.append(X_train[i][0])
    train_label.append(X_train[i][1])


# In[ ]:


count=0
t=[0]*7
for i in range(len(test_data)):
    t[test_label[i]]+=1
print(t)


# In[ ]:


np.sum(t)


# In[30]:


# Cross Validation

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import csv
from xgboost import XGBClassifier


# In[ ]:


# clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial')
# clf = GaussianNB()    


# In[ ]:


main_data_set=train_data
for i in range(len(train_data1)):
    main_data_set[i]+=train_data1[i]

train_data=main_data_set


# In[ ]:


main_data_set1=test_data
for i in range(len(test_data1)):
    main_data_set1[i]+=test_data1[i]


# In[ ]:


main_data_set=train_data1
main_data_set1=test_data1


# In[ ]:


test_data=np.hstack((main_data_set1,test_data1))
train_data=np.hstack((main_data_set,train_data1))


# In[ ]:


test_data=main_data_set1


# In[ ]:


pca = PCA(n_components=500)
train_data1=pca.fit_transform(np.array(train_data))
test_data1=pca.transform(np.array(test_data))


# In[ ]:


np.array(main_data_set1).shape


# In[31]:


X_train, X_test, y_train = np.array(train_data1),np.array(test_data1),np.array(train_label)


# In[ ]:


# train_data1=copy.deepcopy(train_data)
# test_data1=copy.deepcopy(test_data)


# In[32]:


kf=KFold(n_splits=5)
model=[]
score_set1=[]
for train_index, test_index in kf.split(X_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    valid_train_data, valid_test_data = X_train[train_index], X_train[test_index]
    valid_train_label, valid_test_label = y_train[train_index], y_train[test_index]
    clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial')
    clf.fit(np.array(valid_train_data),np.array(valid_train_label))
    b=clf.predict(np.array(valid_test_data))
    score1=clf.score(valid_test_data,valid_test_label)
    print("Accuracy ",score1)
    score_set1.append(score1)
    model.append(clf) 


# In[ ]:


score_mean=np.mean(score_set1)
standard_dev=np.std(score_set1)
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set1)
best=model[np.argmax(score_set1)]
best_score6=best.score(np.array(test_data1),np.array(test_label))
prob_dist6=best.predict_proba(test_data1)
label_pred=best.predict(test_data1).tolist()
print("Test Accuracy by best model in cross validation : ",best_score6)


# # SVM

# In[ ]:


# clf = svm.SVC(gamma=0.001)
# clf.fit(np.array(train_data), np.array(train_label))
# label_predict=clf.predict(np.array(test_data)).tolist()
# score=clf.score(np.array(test_data),np.array(test_label))
# print("Accuracy : ",score)
# prob_dist=clf.predict_proba(np.array(test_data)).tolist()
# prob_dist=np.transpose(prob_dist)


# In[ ]:


from sklearn.svm import LinearSVC


# In[ ]:


model,score_set=cross_validation_svm(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0]
for i in range(7):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_pred[i]]=confusionmatrix[test_label[i]][label_pred[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# ROC Curve 
prob_dist=np.transpose(prob_dist6)
for i in range(7):
    tpr,fpr=roc_design(prob_dist[i],test_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i+1))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 1 to 7")
plt.show()


# In[ ]:


# # ROC Curve 
# # prob_dist=np.transpose(prob_dist)
# for i in range(7):
#     print(i)
#     tpr,fpr=roc_design(prob_dist[i],test_label,i)
#     plt.plot(fpr, tpr ,label="Class"+str(i+1))
# # roccurve(prob_dist,test_label)
# plt.xlabel("False +ve Rate")
# plt.ylabel("True +ve Rate")
# plt.legend()
# plt.title("ROC Curve for Class 1 to 11")
# plt.show()


# In[ ]:


train_data=train_data.tolist()
test_data=test_data.tolist()


# # Naive Bayes

# In[ ]:


model,score_set=cross_validation_gaussian(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# In[ ]:


# # clf = GaussianNB().fit(copy.deepcopy(np.array(train_data)), copy.deepcopy(np.array(train_label)))

# label_predict=clf.predict(np.array(test_data)).tolist()
# score=clf.score(np.array(test_data),np.array(test_label))
# print("Accuracy : ",score)
# prob_dist=clf.predict_proba(np.array(test_data)).tolist()
# prob_dist=np.transpose(prob_dist)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0]
for i in range(7):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_pred[i]]=confusionmatrix[test_label[i]][label_pred[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# ROC Curve 
prob_dist=np.transpose(prob_dist)
for i in range(7):
    tpr,fpr=roc_design(prob_dist[i],test_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i+1))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 1 to 7")
plt.show()


# In[ ]:


# # ROC Curve 
# # prob_dist=np.transpose(prob_dist)
# for i in range(7):
#     print(i)
#     tpr,fpr=roc_design(prob_dist[i],test_label,i)
#     plt.plot(fpr, tpr ,label="Class"+str(i+1))
# # roccurve(prob_dist,test_label)
# plt.xlabel("False +ve Rate")
# plt.ylabel("True +ve Rate")
# plt.legend()
# plt.title("ROC Curve for Class 1 to 11")
# plt.show()


# # Neural Network Model

# In[ ]:


x_train=np.array(train_data)
y_train=np.array(train_label)
x_test=np.array(test_data)
y_test=np.array(test_label)


# In[ ]:


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.models.Sequential()


# In[ ]:


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))


# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train, epochs=50)


# In[ ]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


# # Logistic Regression

# In[ ]:


model,score_set=cross_validation_logistic(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0]
for i in range(7):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_pred[i]]=confusionmatrix[test_label[i]][label_pred[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# ROC Curve 
prob_dist=np.transpose(prob_dist)
for i in range(7):
    tpr,fpr=roc_design(prob_dist[i],test_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i+1))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 1 to 7")
plt.show()


# In[ ]:


# clf = LogisticRegressionCV().fit(copy.deepcopy(np.array(train_data)), copy.deepcopy(np.array(train_label)))
# label_predict=clf.predict(np.array(test_data)).tolist()
# score=clf.score(np.array(test_data),np.array(test_label))
# print("Accuracy : ",score)


# In[ ]:


prob_dist=clf.predictproba(np.array(test_data),np.array(test_label)).tolist()
prob_dist=np.transpose(prob_dist)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0]
for i in range(7):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_predict[i]]=confusionmatrix[test_label[i]][label_predict[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# ROC Curve 
# prob_dist=np.transpose(prob_dist)
for i in range(7):
    print(i)
    tpr,fpr=roc_design(prob_dist[i],test_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i+1))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 1 to 11")
plt.show()


# # Enseamble Learning

# In[ ]:


def adaboost(n,train_data1,train_label1,test_data,test_label,weights,d):
    
    alpha_k=[]
    Ck=[]
    nat=[i for i in range(len(train_data1))]
    main_data=copy.deepcopy(train_data1)
    main_label=copy.deepcopy(train_label1)
    
    for i in range(n):
        print(i)
#         print("Hello : ",i)
        sample = choice(nat, d,p=weights,replace=False)
#         sample = choice(nat,d,weights,replace=False)
        train_data=[]
        train_label=[]
        for j in range(len(sample)):
            train_data.append(train_data1[sample[j]])
            train_label.append(train_label1[sample[j]])
        
        clf=DecisionTreeClassifier(max_depth=3,max_leaf_nodes=10)
        clf.fit(np.array(train_data),np.array(train_label))
        predict1=clf.predict(np.array(main_data))
        h=labelling(predict1.tolist(),main_label)
        train_err=clf.score(np.array(train_data),np.array(train_label))
        train_err=1-train_err
        alpha=0.5*np.log((1-train_err)/float(train_err))+np.log(25)
        alpha_k.append(alpha)
        Ck.append(clf)
#         print("Hello1 : ",i)
        for j in range(len(weights)):
            
            if h[j]==1:
                
                weights[j]=weights[j]*math.exp((-1)*alpha)
            else:
                weights[j]=weights[j]*math.exp(alpha)
        w=copy.deepcopy(weights)
        total=np.sum(w)
        for j in range(len(weights)):
            weights[j]=weights[j]/float(total)
#         print("Hello2 : ",i)
 #For test set
    test_predict=[]
    for i in range(len(test_data)):
        disc_func=[[] for i in class_label]
        for j in range(k_max):
            index=Ck[j].predict(np.array(test_data[i]).reshape(1,-1)).tolist()[0]
            if disc_func[index]==[]:
                disc_func[index].append(alpha_k[j])
            else:
                disc_func[index][0]+=alpha_k[j]
        test_predict.append(disc_func.index(max(disc_func)))

    test_acc=accuracy(test_predict,test_label)   
# For train set
    train_predict=[]
#     for i in range(len(train_data1)):
#         disc_func1=[[] for i in class_label]
#         for j in range(k_max):
#             index=Ck[j].predict(np.array(train_data1[i]).reshape(1,-1)).tolist()[0]
#             if disc_func1[index]==[]:
#                 disc_func1[index].append(alpha_k[j])
#             else:
#                 disc_func1[index][0]+=alpha_k[j]
#         train_predict.append(disc_func1.index(max(disc_func1)))

#     train_acc=accuracy(train_predict,train_label1)  
    train_acc=0
    return Ck,alpha_k,train_predict,test_predict,train_acc,test_acc
            


# In[ ]:


k_max=50
d=3000
class_label=list(set(train_label))


# In[ ]:


weights=[1/float(len(train_data)) for i in train_data]
Ck,alpha_k,train_result,test_result,train_acc,test_acc=adaboost(k_max,train_data,train_label,test_data,test_label,weights,d)


# In[ ]:


print("Accuracy in training : ",train_acc)
print("Accuracy in testing : ",test_acc)


# # Histogram as Features

# In[ ]:


def histogram(train_data):
    hist=[0]*256
    for i in range(len(train_data)):
        hist[train_data[i]]+=1
    return hist
        


# In[ ]:


train_data1=[]
for i in range(len(train_data)):
    train_data1.append(histogram(train_data[i]))
test_data1=[]
for i in range(len(test_data)):
    test_data1.append(histogram(test_data[i]))


# In[ ]:


train_data=copy.deepcopy(train_data1)
test_data=copy.deepcopy(test_data1)


# # Logistic Regression

# In[ ]:


model,score_set=cross_validation_logistic(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0]
for i in range(7):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_pred[i]]=confusionmatrix[test_label[i]][label_pred[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# ROC Curve 
prob_dist=np.transpose(prob_dist)
for i in range(7):
    print(i)
    tpr,fpr=roc_design(prob_dist[i],test_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i+1))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 1 to 11")
plt.show()


# # Naive Bayes

# In[ ]:


model,score_set=cross_validation_gaussian(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0]
for i in range(7):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_pred[i]]=confusionmatrix[test_label[i]][label_pred[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# ROC Curve 
prob_dist=np.transpose(prob_dist)
for i in range(7):
    print(i)
    tpr,fpr=roc_design(prob_dist[i],test_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i+1))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 1 to 11")
plt.show()


# # SVM

# In[ ]:


model,score_set=cross_validation_svm(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0]
for i in range(7):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_pred[i]]=confusionmatrix[test_label[i]][label_pred[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# ROC Curve 
prob_dist=np.transpose(prob_dist)
for i in range(7):
    print(i)
    tpr,fpr=roc_design(prob_dist[i],test_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i+1))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 1 to 11")
plt.show()


# # Neural Network

# In[ ]:


x_train=np.array(train_data)
y_train=np.array(train_label)
x_test=np.array(test_data)
y_test=np.array(test_label)


# In[ ]:


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.models.Sequential()


# In[ ]:


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))


# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train, epochs=50)


# In[ ]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


# # Adaboosting

# In[ ]:


k_max=50
d=3000
class_label=list(set(train_label))


# In[ ]:


def accuracy(predict,true):
    count=0
    for i in range(len(predict)):
        if predict[i]==true[i]:
            count+=1
    return count/float(len(predict))


# In[ ]:


weights=[1/float(len(train_data)) for i in train_data]
Ck,alpha_k,train_result,test_result,train_acc,test_acc=adaboost(k_max,train_data,train_label,test_data,test_label,weights,d)


# In[ ]:


print("Accuracy in training : ",train_acc)
print("Accuracy in testing : ",test_acc)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0]
for i in range(7):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][test_result[i]]=confusionmatrix[test_label[i]][test_result[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


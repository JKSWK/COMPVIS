#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')


# Wczytuję zbiór danych

# In[8]:


df = pd.read_csv("age_gender.csv")


# In[27]:


df.head().info


# Od razu zwracam uwagę na fakt, że wartości w kolumnie "pixels" są w dość dziwnym formacie. Wykażę to w poniższych komórkach.

# In[30]:


df.sample(5)


# In[32]:


df.dtypes


# In[33]:


df.pixels


# In[34]:


df.pixels[0]


# Wartości kolumny pixels, to reprezentacje 8 bitowe kolorów w zakresie 0-255, zapisane jako 23705 objektów string.

# In[36]:


#df.drop('img_name', axis=1, inplace=True)


# In[40]:


df.shape


# In[41]:


df.isnull().sum()

Zbiór danych nie wykazuje wartości brakujących
# In[44]:


df['age'].value_counts()


# In[45]:


df['ethnicity'].value_counts()


# In[46]:


df['gender'].value_counts()


# Pierwszym problemem związanym z tym zbiorem danych jest ich niezbalansowanie w zmiennych "age" i "ethnicity". Najliczniejsze kategorie są ponad dwukrotnie większe niż te na drugim miejscu itd.

# In[47]:


columns = ["age", "gender", "ethnicity"]
y = df.drop("pixels", axis=1)
X = df.drop(columns, axis=1)


# In[48]:


X.head()


# In[49]:


for i in y.columns:
    plt.figure(figsize=(15,7))
    g = sns.countplot(x = y[i], palette="icefire")
    plt.title(f"Number of {i}")


# In[50]:


y["age"] = pd.cut(y["age"],bins=[0,25,45,116],labels=["0-25","25-45","45-116"])
plt.figure(figsize=(15,7))
g = sns.countplot(x = y["age"], palette="icefire")
plt.title("Number of age")


# In[52]:


len(X["pixels"][0].split(" "))


# In[53]:


np.sqrt(len(X["pixels"][0].split(" ")))


# Każda wartość w kolumnie to 2304 reprezentacji 8-bitowych. Tworzą one kwadraty pixeli o rozmiarze 48 x 48.

# In[54]:


X = (
    X['pixels'].str.split(' ', expand=True)
        .astype(int).to_numpy()
        .reshape((23705, 48, 48, 1))
)


# In[55]:


from sklearn.model_selection import train_test_split


# ## Wyświetlenie próbek zdjęć, zagregowanych po kloumnie "ethnicity".
# Jak widać, zdarzają się przypadki, gdzie poprawność klasyfikacji pod względem rasy, jest obiektywnie mówiąc błędna. Należy to wziąć pod uwagę przy badaniu predykcji modelu.

# In[56]:


plt.figure(figsize=(16,16))
for i,a in zip(df.loc[df.ethnicity == 0].index.to_list()[1000:1026], range(1,26)):
    plt.subplot(5,5,a)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i])
    plt.xlabel(
    "Age: "+str(y['age'].iloc[i])+
    " Ethnicity:"+str(y['ethnicity'].iloc[i])+
    " Gender:"+str(y['gender'].iloc[i]))
plt.show()


# In[57]:


plt.figure(figsize=(16,16))
for i,a in zip(df.loc[df.ethnicity == 1].index.to_list()[1000:1026], range(1,26)):
    plt.subplot(5,5,a)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i])
    plt.xlabel(
    "Age: "+str(y['age'].iloc[i])+
    " Ethnicity:"+str(y['ethnicity'].iloc[i])+
    " Gender:"+str(y['gender'].iloc[i]))
plt.show()


# In[58]:


plt.figure(figsize=(16,16))
for i,a in zip(df.loc[df.ethnicity == 2].index.to_list()[1000:1026], range(1,26)):
    plt.subplot(5,5,a)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i])
    plt.xlabel(
    "Age: "+str(y['age'].iloc[i])+
    " Ethnicity:"+str(y['ethnicity'].iloc[i])+
    " Gender:"+str(y['gender'].iloc[i]))
plt.show()


# In[59]:


plt.figure(figsize=(16,16))
for i,a in zip(df.loc[df.ethnicity == 3].index.to_list()[1000:1026], range(1,26)):
    plt.subplot(5,5,a)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i])
    plt.xlabel(
    "Age: "+str(y['age'].iloc[i])+
    " Ethnicity:"+str(y['ethnicity'].iloc[i])+
    " Gender:"+str(y['gender'].iloc[i]))
plt.show()


# In[60]:


plt.figure(figsize=(16,16))
for i,a in zip(df.loc[df.ethnicity == 4].index.to_list()[1000:1026], range(1,26)):
    plt.subplot(5,5,a)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i])
    plt.xlabel(
    "Age: "+str(y['age'].iloc[i])+
    " Ethnicity:"+str(y['ethnicity'].iloc[i])+
    " Gender:"+str(y['gender'].iloc[i]))
plt.show()


# In[61]:


y.ethnicity = y.ethnicity.replace([0,1,2,3,4],["White","Black","Asian","Hindu","Latin"])
y.gender = y.gender.replace([0,1],["Male","Female"])
# y = pd.get_dummies(data=y, columns=['age', "ethnicity", "gender"])
# y = y.astype(float)


# In[62]:


#TODO list of labels, each element of list is list of [age,ethnictity,gender]

labels = []

for idx in range(X.shape[0]):
  labels.append(list(y.loc[idx].values))
  
print(labels[:10])


# #Training

# In[94]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


# In[96]:


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
# from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
#from imutils import paths
import tensorflow as tf
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# In[115]:


# 0-255 to 0-1
X_train = X_train/255 
X_test = X_test/255

# Change the labels from integer to categorical data
train_y_one_hot = tf.keras.utils.to_categorical(y_train) 
test_y_one_hot = tf.keras.utils.to_categorical(y_test)


# ## PIERWSZY MODEL

# In[ ]:


#definicja modelu CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", input_shape=(48, 48, 1)))
model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
model.add(Dense(1, activation="sigmoid"))

# compile model

model.compile(optimizer="adam", loss= "binary_crossentropy", metrics=['accuracy'])

# fit model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), verbose=1)


# ## Model jest przeuczony

# In[83]:


get_ipython().run_line_magic('matplotlib', 'inline')
def plot_history(history):
    #Plot the Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    #Plot the Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
plot_history(history)


# In[84]:


get_ipython().system('pip install imutils')


# Tutaj będę dokonywać próby klasyfikacji wieku, rasy i płci dla zdjęć dostarczonych przeze mnie

# In[85]:


## PREDYKCJA TWARZY
__
#  bapcia.jpg
#  cho.jpg
#  ja.jpg
#  jackson.jpg
#  kylian.jpg
#  mariusz.jpg
#  merkel.jpg
#  son.png
#  sunak.jpg


# In[86]:


image = cv2.imread("cho.jpg")
import imutils
output = imutils.resize(image, width=400)
 
# pre-process the image for classification

image = cv2.resize(image, (48, 48))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


# In[87]:


proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]


# Wyniki w formule top 3 wyników na podstwie prawdopodobieństwa pozostawiają wiele do życzenia

# In[88]:


for (i, j) in enumerate(idxs):
	# build the label and draw the label on the image
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)


# Natomiast prawdopodobieństwo ukazane w globalnym rankingu jest bardziej precyzyjne

# ## DRUGI MODEL Z ELEMENTAMI ZAPOBIEGAJĄCYMI PRZEUCZENIU.

# In[89]:


#definicja modelu CNN z elementami zapobiegającymi przeuczeniu
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

model2 = Sequential()

model2.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", input_shape=(48, 48, 1)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.1))

model2.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model2.add(BatchNormalization())
model2.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.15))

model2.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model2.add(BatchNormalization())
model2.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.20))

model2.add(Flatten())
model2.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
model2.add(BatchNormalization())
model2.add(Dropout(0.25))
model2.add(Dense(10, activation="sigmoid"))

# compile model

model2.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# fit model
history2 = model2.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), verbose=1)


# ## MODEL JEST LEPIEJ DOUCZONY NIŻ PIERWSZY MODEL, ALE IMPLIKACJE TEGO FAKTU MAJĄ MIEJSCE PÓŹNIEJ

# In[90]:


get_ipython().run_line_magic('matplotlib', 'inline')
def plot_history(history):
    #Plot the Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    #Plot the Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b',linewidth=3.0)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
plot_history(history2)


# In[141]:


## PREDYKCJA TWARZY
__
#  bapcia.jpg
#  cho.jpg
#  ja.jpg
#  jackson.jpg
#  kylian.jpg
#  mariusz.jpg
#  merkel.jpg
#  son.png
#  sunak.jpg


# Ponownie weryfikuje model moimi zdjęciami

# In[1]:


image = cv2.imread("mariusz.jpg")
import imutils
output = imutils.resize(image, width=400)
 
# pre-process the image for classification

image = cv2.resize(image, (48, 48))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


# In[92]:


proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]


# In[93]:


for (i, j) in enumerate(idxs):
	# build the label and draw the label on the image
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)


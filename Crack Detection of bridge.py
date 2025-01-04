#!/usr/bin/env python
# coding: utf-8

# In[33]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[16]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report


# In[17]:


positive_dir = Path('/kaggle/input/po-and-negat/positive')
negative_dir = Path('/kaggle/input/po-and-negat/negative')


# In[20]:


file_list = list(positive_dir.glob('*'))
file_list = list(negative_dir.glob('*'))


# In[22]:


def generate_df(image_dir, label):
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df


# In[23]:


positive_df = generate_df(positive_dir, label="POSITIVE")
negative_df = generate_df(negative_dir, label="NEGATIVE")

all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
all_df


# In[24]:





# In[33]:


from sklearn.model_selection import train_test_split
import tensorflow as tf


train_df, temp_df = train_test_split(
    all_df.sample(999, random_state=1),
    train_size=0.7,
    shuffle=True,
    random_state=1
)

test_df, validation_df = train_test_split(
    temp_df,
    test_size=0.5,  
    shuffle=True,
    random_state=1
)


train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

validation_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)


# In[104]:


train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
   rescale=1./255,
   validation_split=0.2
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
   rescale=1./255
)


# In[118]:


train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(55, 55),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)
val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(55, 55),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)
test_data = train_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(55, 55),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False,
    seed=42
)


# In[119]:


inputs = Input(shape=(220, 220, 3))

x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)


outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

optimizer = Adam(learning_rate=0.001) 
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


# In[140]:


y_pred = model.predict(test_data)
y_pred_binary = (y_pred > 0.5).astype(int)
y_true = test_data.classes

f1 = f1_score(y_true, y_pred_binary)
print("Test Set F1 Score:", f1)


# In[153]:


from sklearn.metrics import precision_score, recall_score

y_pred = model.predict(test_data)
y_pred_binary = (y_pred > 0.5).astype(int)  
y_true = test_data.classes
 
precision = precision_score(y_true, y_pred_binary, zero_division=1)
recall = recall_score(y_true, y_pred_binary)

print("Precision:", precision)
print("Recall:", recall)


# In[144]:


from sklearn.metrics import precision_recall_curve, auc
y_pred = model.predict(test_data)
y_true = test_data.classes
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
auc_pr = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AUC = %0.2f)' % auc_pr)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower left')
plt.show()


# In[124]:


history = model.fit(
    train_datagen.flow_from_dataframe(
        train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(55,55),
        batch_size=32,
        class_mode='binary',
        shuffle=True,
        seed=42,
        subset='training'
    ),
    validation_data=val_data,
    epochs=10, 
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, 
            restore_best_weights=True
        )
    ]
)


# In[125]:


plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# In[116]:


fig = px.line(
    history.history,
    y=['loss', 'val_loss'],
    labels={'index': "Epoch", 'value': "Loss"},
    title="Training and Validation Loss Over Time"
)

fig.show()


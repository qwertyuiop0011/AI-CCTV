import warnings
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
import os
import numpy as np
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')


train_x = []
train_y = []
img_list = []
for path in os.listdir('/Users/leejeesung/Documents/newDataset/000000'):
    img = Image.open("/Users/leejeesung/Documents/newDataset/000000/"+path)
    train_x.append(np.array(img))
    train_y.append(1)
    img.close()
for path in os.listdir('/Users/leejeesung/Documents/newDataset_normal/000000'):
    img = Image.open("/Users/leejeesung/Documents/newDataset_normal/000000/"+path)
    train_x.append(np.array(img))
    train_y.append(0)
    img.close()

train_x = np.array(train_x)
train_x  = train_x.astype(np.float32)
train_x /= 255.0
train_y = np.array(train_y)

#strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = 64

#STEP_SIZE = 1000000

base_model = ResNet50(include_top = False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation = 'relu')
x = Dense(256, activation = 'relu')
pred = Dense(1, activation='sigmoid')(x)
model = Model(inputs = base_model.input, outputs=pred)
opt = SGD(lr=0.0001, momentum = 0.9)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])

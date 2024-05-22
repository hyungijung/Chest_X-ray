from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.metrics import Accuracy, Precision, Recall

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import os

def unet(input_size=(224, 224, 1)):
    inputs = Input(input_size)

    # Encoding path
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoding path
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

unet = unet()

def preprocess(x):
    # 양방향 필터
    filtered_image = cv2.bilateralFilter(x, d=9, sigmaColor=75, sigmaSpace=75)

    # unsharp
    gaussian = cv2.GaussianBlur(filtered_image, (9,9), 10.0)
    img_sharpened = cv2.addWeighted(filtered_image, 1.5, gaussian, -0.5, 0)

    # 리사이즈
    img_resized = cv2.resize(img_sharpened, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    # 히스토그램 평탄화
    img_equalized = cv2.equalizeHist(img_resized)

    # 정규화, 차원 추가
    img_normalized = np.expand_dims(img_equalized, axis=-1) / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)

    return img_normalized


paths = ['/content/drive/MyDrive/Colab Notebooks/VGG16_input/normal',
         '/content/drive/MyDrive/Colab Notebooks/VGG16_input/tb',
         '/content/drive/MyDrive/Colab Notebooks/VGG16_input/pn']
x = []
y = []

for path in paths:
  for filename in os.listdir(path):
      if filename.endswith(('.jpeg', 'jpg', '.png')):
          img_path = os.path.join(path, filename)
          img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
          img = preprocess(img)
          if img is not None:
              x.append(img)
              if path.endswith('normal'):
                  y.append(0)
              elif path.endswith('tb'):
                  y.append(1)
              elif path.endswith('pn'):
                  y.append(2)


del paths
del img
del img_path

OHE = OneHotEncoder(sparse=False)

x = np.array(x)
y = np.array(y)

y = y.reshape(-1, 1)
y = OHE.fit_transform(y)

print(OHE.categories_)


print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential([
    VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),

    Flatten(),

    # 1
    Dense(512, kernel_initializer=HeNormal()),
    LeakyReLU(alpha=0.01),  # LeakyReLU 활성화 함수 적용, alpha 값 조정 가능
    BatchNormalization(),   # 배치 정규화 적용
    Dropout(0.6),           # 드롭아웃 적용, 과적합 방지

    # 2
    Dense(256, kernel_initializer=HeNormal()),
    LeakyReLU(alpha=0.01),  # LeakyReLU 활성화 함수 적용
    BatchNormalization(),   # 배치 정규화 적용
    Dropout(0.6),           # 드롭아웃 적용

    # output
    Dense(3, activation='softmax')  # softmax 활성화 함수를 사용한 분류
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size= 32)

model.evaluate(x_test, y_test)

model.save('/content/drive/MyDrive/Colab Notebooks/4_17_VGG16_model.keras')


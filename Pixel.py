import numpy as np
import cv2
import os
import tensorflow as tf

from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.utils import to_categorical
from PIL import Image


def load_sampling():

    data = []
    label = []

    tb_path = 'C:/Users/hyun2/PycharmProjects/FinalProject/data/tb_mask'
    pn_path = 'C:/Users/hyun2/PycharmProjects/FinalProject/data/pn_mask'
    normal_path = 'C:/Users/hyun2/PycharmProjects/FinalProject/data/normal_mask'

    for subdir, _, filename in os.walk(tb_path):
        for file in filename:
            image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            data.append(image)
            label.append(os.path.basename(subdir))

    for subdir, _, filename in os.walk(pn_path):
        for file in filename:
            image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            data.append(image)
            label.append(os.path.basename(subdir))

    for subdir, _, filename in os.walk(normal_path):
        for file in filename:
            image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            data.append(image)
            label.append(os.path.basename(subdir))


    data = np.array(data)
    print(data.shape)

    # 문자열을 숫자로 매핑하는 딕셔너리
    label_mapping = {'tb_mask': 0, 'pn_mask': 1, 'normal_mask': 2}
    # 문자열 레이블을 숫자로 변환
    label = np.array([label_mapping[label] for label in label])



    # 이미지 데이터를 2차원으로 변환
    n_samples, height, width, channels = data.shape
    images_reshaped = data.reshape((n_samples, height * width * channels))

    # 언더샘플링을 수행합니다.
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(images_reshaped, label)



    X_res_reshaped = X_res.reshape((-1, height, width, channels))


    y_train_one_hot = to_categorical(y_res, num_classes=3)
    print(X_res_reshaped.shape)
    print(y_train_one_hot.shape)

    return X_res_reshaped, y_train_one_hot
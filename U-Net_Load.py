from tensorflow import keras
import cv2
import numpy as np
import os

model = keras.models.load_model('C:/Users/hyun2/PycharmProjects/FinalProject/U-Net.keras')


def preprocess_and_extract_roi(image_path, output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 양방향 필터
    filtered_image = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

    # 마스크 생성: 원본 이미지에서 흐릿한 이미지 빼기
    mask = cv2.subtract(image, filtered_image)
    # 세부 사항 강조: 원본 이미지에 마스크 더하기
    image = cv2.add(image, mask)

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)  # 모델에 맞는 크기로 조정

    image_ori = image
    # 대비 조절
    contrast = 1.5  # 예시 값 (1보다 클 경우 대비 증가)
    image_ori = cv2.multiply(image_ori, np.array([contrast]).astype('uint8'))

    image = cv2.equalizeHist(image)

    # image_ori = image
    image = image / 255.0  # 정규화

    # 이미지를 모델에 입력하기 전에 차원을 조정
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    image = np.expand_dims(image, axis=-1)  # 채널 차원 추가

    image = predict_and_save(model, image, image_ori)
    #결과 이미지 저장
    save_image(image_path, "Mask", image, output_folder)



def predict_and_save(model, img, img_ori):
    image = img
    prediction = model.predict(image)

    # 원본 이미지 복구(전처리 된 이미지)
    img = (img * 255).astype(np.uint8)  # 정규화된 값을 다시 [0, 255] 범위로 변환

    img = np.squeeze(img, axis=0)  # 배치 차원 제거 -> (height, width, 1)
    img = np.squeeze(img, axis=-1)  # 채널 차원 제거 -> (height, width)

    # 모델에서 출력된 이미지를 복구
    prediction = (prediction * 255).astype(np.uint8)  # 정규화된 값을 다시 [0, 255] 범위로 변환

    image_restored = np.squeeze(prediction, axis=0)  # 배치 차원 제거 -> (height, width, 1)
    image_restored = np.squeeze(image_restored, axis=-1)  # 채널 차원 제거 -> (height, width)

    # ****************************
    # 모델에서 출력된 이미지 후처리

    # 대비 조절
    contrast = 3.0  # 예시 값 (1보다 클 경우 대비 증가)
    image_contrast = cv2.multiply(image_restored, np.array([contrast]).astype('uint8'))


    # 임계값 설정
    threshold = 127  # 임계값은 0에서 255 사이의 값
    # 임계값을 기준으로 픽셀 값 조건부 변경
    # 픽셀 값이 임계값을 넘으면 255, 넘지 않으면 0
    ret, thresholded_image = cv2.threshold(image_contrast, threshold, 255, cv2.THRESH_BINARY)

    # ****************************

    # 마스크
    roi = cv2.bitwise_and(img_ori, img_ori, mask=thresholded_image)

    return roi

def save_image(original_path, suffix, image, output_folder):

    # 파일 이름과 확장자 분리
    filename = os.path.basename(original_path)
    filename_without_ext = os.path.splitext(filename)[0]
    new_filename = f"{filename_without_ext}_{suffix}.png"
    save_path = os.path.join(output_folder, new_filename)

    # 이미지 파일 저장
    cv2.imwrite(save_path, image)
    print(f"Image saved to {save_path}")





# 데이터 폴더와 결과 폴더 설정
data_folder = 'C:/Users/hyun2/PycharmProjects/FinalProject/data/normal'
output_folder = 'C:/Users/hyun2/PycharmProjects/FinalProject/data/normal_mask2'


# 결과 폴더 생성 (존재하지 않을 경우)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 데이터 폴더 내 모든 이미지에 대해 처리
for filename in os.listdir(data_folder):
    if filename.endswith(('.jpeg', '.png')):
        image_path = os.path.join(data_folder, filename)
        preprocess_and_extract_roi(image_path, output_folder)

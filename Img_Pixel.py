"""
    7강 이미지 픽셀조작 및 ROI

        이미지의 임의의 픽셀 값을 취하여 그 값을 수정하기
        이미지 속성 얻기
        이미지 영역 설정하기
        이미지 채녈을 분할하고 합치기
"""

try:
    from numpy import numpy as np
    from cv2 import  cv2
except ImportError:
    pass

img_url = 'img/100.jpg'
img = cv2.imread(img_url)
px = img[340, 200]
# 이미지의 340, 200의 pixle 값을 저장[B G R]
# 필셀값을 검정색으로 변경 하려면,  img[340, 200] = [0, 0, 0]
# 하지만 성능상에 문제가 있을 수 있음 이에 더 최적화된 알고리즘이 적용된 Numpy를 사용
# B = img.item(340, 200, 0)
# G = img.item(340, 200, 0)
# R = img.item(340, 200, 0)
# BGR = [B, G, R]


print(px)
# 출력 [89 91 99]

# 이미지 속성 얻기 위해 Numpy array에 접근
#   img.shape : 이미지 해상도 및 컬러 채널(Image Height,  Image width, 채널)
#   img.size : 이미지 크기(바이트)
#   img.dtype : 이미지 데이터 타입

print(img.shape)
print(img.size)
print(img.dtype)

# ROI -> Numpy 인텍싱을 통해 얻을 수 있음
cv2.imshow('original', img)

subimg = img[300:400, 320:720]
# 원본이미지의 Y축 300 ~ 400, X축 320 ~ 720 -> 400 x 100 크기의 이미지
cv2.imshow('cutting', subimg)

img[300:400, 0:400] = subimg

print("원본=", img.shape)
print("subimg = ", subimg.shape)

cv2.imshow('modified', img)

# 이미지 채널 분할 및 합치기

b, g, r = cv2.split(img)
# img를 B, G, R 채널별로 분리한 후, r, g, b에 저장
# split 함수는 성능면에서 효율적인 함수가 아님 따라서 꼭 필요한 경우에만 사용
# Numpy 인덱싱을 사용하는 것이 효율적임.
# b = img[:, :, 0]
# g = img[:, :, 1]
# r = img[:, :, 2]
# 이미지의 모든 픽셀의 Red 값을 0으로 만들고자 하면 다음과 같이...
# img[:, :, 2] = 0


print("img[100, 100] = ", img[100, 100])
#img[100, 100] = [233 240 243]

print("b[100, 100] =",  b[100, 100])
#b[100, 100] = 233

print("g[100, 100] =",  g[100, 100])
#g[100, 100] = 240

print("r[100, 100] =",  r[100, 100])
#r[100, 100] = 243

cv2.imshow('blue channel', b)
cv2.imshow('green channel', g)
cv2.imshow('red channel', r)
# 위 3가지 모두 흑백으로 보임

# cv2.merge() 함수를 이용하면 채널을 합쳐 컬러 이미지로 만들 수 있음
merged_img = cv2.merge((b, g, r))
cv2.imshow('merged img', merged_img)

cv2.waitKey(0)
cv2.destroyAllWindows( )

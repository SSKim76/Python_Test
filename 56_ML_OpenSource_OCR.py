"""
56강. 머신러닝 기초3 - OpenSource를 이용한 OCR(Tesseract-OCR)

    Tesseract는 OCR을 위한 엔진.
    2006년 구글의 지원하에 개발되고 있는 무료 소프트웨어
    아파치 라이선스로 자유롭게 활용가능.
    Trsseract는 통용되는 OCR 오픈소스 중 가장 정확도가 높다고 알려져 있음.

    1. 사전작업
        1-1. pip install pillow
            파이썬 PIL(Python Imaging Library) 라이브러리의 수정 배포판
        1-2. pip install pytesseract
        1-3. Tesseract-OCR의 윈도우용 설치파일 설치
            단독 프로그램이라 프로그램을 따로 실행해서 OCR을 수행 할 수 있음.
        1-4. cmd -> tesseract 실행하여 정상적으로 호출 되는지 확인
        1-5. Python Console에서 import Test
            >>> import PIL
            >>> import pytesseract

    2. Test Image
        1-1. OCR_ENG.jpg : OCR Test용 영문 이미지
        1-2. OCR_KOR.jpg : OCR Test용 한글 이미지

    3. 실행
        1-1. OCR을 수행할 이미지를 PIL의 Image객체로 생성
        1-2. pytesser의 image_to_string()을 이용해 Tesseract의 OCR기능을 호출하여 결과를 얻는다.

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef

from PIL import Image
from pytesseract import *


def OCR(imgfile, lang = 'eng'):
    global resultFile

    im = Image.open(imgfile)
    text = image_to_string(im, lang = lang)

    print('++++++++++ OCR Result ++++++++++')
    print(text)

    f = open(resultFile, mode = 'wt', encoding = 'utf-8')
    f.write(text)
    f.close()
# End of def OCR()


engImg = 'img/OCR_ENG.jpg'
korImg = 'img/OCR_KOR.png'
Img = 'img/opencv_logo.jpg'
resultFile = 'img/OCR_Img/result.txt'

#img = cv2.imread(korImg, cv2.IMREAD_COLOR)
#cv2.imshow('view',img)
#impDef.close_window( )

#OCR(engImg, 'eng')
OCR(korImg, 'kor')
#OCR('img/OCR_Img/8.jpg')
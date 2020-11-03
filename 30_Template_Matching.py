"""
30강. Template Matching

    템플릿 매칭
        어떤 이미지에서 부분 이미지를 검색하고 찾는 방법.
        템플릿 : 부분이미지

    원리
        템플릿 이미지의 중심을(x,y)라고 할때, 템플릿 이미지를 타겟 이미지 위에 두고 템플릿 이미지로 덮인
        타겟이미지 부분의 픽셀값과 템플릿 이미지의 픽셀값을 특정 수학 연산으로 비교한다. 이 값을 R(x,y)일때
        타겟이미지 전체를 미끄러져 가면서 비교한 결과인 R(x,y) 전체는 타겟 이미지 보다 작은 이미지가 된다.
        타겟 이미지의 사이즈가 W x H이고, 템플릿 이미지의 사이즈가 w x h 라면,
        결과 이미지는 W-w+1 x H-h+1의 크기가 된다.

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef



# 이미지에서 템플릿 이미지와 비교하여 템플릿이미지와 가장 매칭되는 부분을 사각형으로 표시한다.
def tmpMatch(ImgNo, tempNo):
    img = cv2.imread(impDef.select_img(ImgNo))
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = img1.copy()

    template = cv2.imread(impDef.select_img(tempNo), cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]

    methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img1 = img2.copy()
        method = eval(meth)

        try:
            res = cv2.matchTemplate(img1, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        except:
            print('Error', meth)
            continue

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0]+w, top_left[1]+h)
        cv2.rectangle(img1, top_left, bottom_right, (255,255,255), 3)

        plt.suptitle(meth)

        plt.subplot(131), plt.imshow(img, cmap ='gray')
        plt.title('input Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(132), plt.imshow(res, cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

        plt.subplot(133), plt.imshow(img1, cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

        plt.show()

tmpMatch(12, 13)
tmpMatch(12, 14)
tmpMatch(12, 15)

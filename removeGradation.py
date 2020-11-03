"""

조명에 의해 생긴 Gradation을 제거하기 위해...

class removeLightGradation():
    1. Check Image Size 
    2. LAB모델을 이용한 Color Space (RGB -> LAB변환)
    3. Median Filter(Radius : 20 ~ 50, 100 : 실제와 가장 근사한 조명상태 구현)
    4. 3번 이미지 반전하여 역조명 채널 생성
    5. 원본영상에 합성
    6. Histogram 최대-최소평균으로 Golbal Thresholding

"""

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import math


class removeLightGradation:

    def __init__(self, img):
        self.img = img
        

    def convertLAB(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)


    def addMedianFilter(self, labImg, val = 55):
        filterImg = cv2.medianBlur(labImg, val)
        return filterImg
       

    def createReverseImg(self, filterImg):
        return cv2.bitwise_not(filterImg)


    def mergeImg(self, img1, img2):
        return cv2.add(img1, img2)
    
    def imgBlending(self, img1, img2, val):
        return cv2.addWeighted(img1, val, img2, 1-val, 0)


    def globalThresholding(self, img):
        #Histogram 의 최대-최소 평균으로 Global Thresholding
        ret, thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr

    def thresholding(self, img, threshold = 127, value = 255):
        #ret, thr9 = cv2.threshold(img, threshold, value, cv2.THRESH_BINARY)
        thr10 = cv2.adaptiveThreshold(img, value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        thr11 = cv2.adaptiveThreshold(img, value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        titles = ['adaptive mean', 'adaptive gaussian']
        images = [thr10, thr11]
        showPlot(titles, images)


def showImg(title, img):
    cv2.imshow(title, img)

    cv2.waitKey(0)
    #k=cv2.waitKey(0) & 0xFF
    # if k == ord('s'):
    cv2.destroyAllWindows()


def showPlot(titles, images):
    
    num = len(titles)
    
    if num % 2 :
        row = int(num/2) + 1
    else:
        row = int(num/2)
    
    #print("num = {1}, row = {0}".format(row, num))
    
    cnt = 0
    for i in range(0,row):
        cnt+=1
        plt.subplot(row, 2, cnt), plt.imshow(images[cnt-1], cmap = 'gray')
        plt.title(titles[cnt-1]), plt.xticks([]), plt.yticks([])
        #print("cnt = {0}, row = {1}, [cnt-1] = {2}".format(cnt, row, cnt-1))

        cnt+=1
        
        try:
            if titles[i+2]:
                plt.subplot(row, 2, cnt), plt.imshow(images[cnt-1], cmap = 'gray')
                plt.title(titles[cnt-1]), plt.xticks([]), plt.yticks([])
                #print("cnt = {0}, row = {1}, [cnt-1] = {2}".format(cnt, row, cnt-1))    
        except:
            pass        
    
    plt.show()


def viewResult2(viewData, row, camp = ''):
#viewData = {'Original': img, 'Gray Img' : img_gray, 'Alpha Img' : img_alp}

    dataLen = len(viewData)
    cols = math.ceil(dataLen/row)
    #print("cols = ", cols)

    i = 1
    for key, val in viewData.items():
        subplotNo = str(cols)+str(row)+str(i)
        #print('key = ', key)
        #print('subplotNo = ', subplotNo)
        #cv2.imshow(key, val)


        if(camp == 'gray') :            
            plt.subplot(subplotNo), plt.imshow(val, cmap = 'gray')
            plt.title(key), plt.xticks([]), plt.yticks([])
        else :
            plt.subplot(subplotNo), plt.imshow(val)
            plt.title(key), plt.xticks([]), plt.yticks([])      

        i = i + 1
    # End of for key, val int viewData.items():
    plt.show()


# End of viewResult2()
# for i in range(1,11):
#     if i % 2 :
#         row = int(i/2) + 1
#     else:
#         row = int(i / 2)
#     #print("int({0} / 2) = {1}, int(round( {0} / 2)) = {2}".format(i, int(i/2), int(round(i/2))))
#     print("i = {0}, row = {1}".format(i, row))


# Open Image - Color Image로 Open 해야 함.... 
img = cv2.imread('gradation03.jpg', cv2.IMREAD_COLOR)
showImg("원본", img)

rt = removeLightGradation(img)

# Change Color Space to LAB 
labImg = rt.convertLAB()
#showImg('LAB', labImg)

# split Channel
l, a, b = cv2.split(labImg)
#showImg('labImg - l', l)
#showImg('labImg - a', a)
#showImg('labImg - b', b)


# add Median Filter
filterImg = rt.addMedianFilter(l, 55)
#showImg('Median 99', filterImg)
labFilterImg = rt.addMedianFilter(labImg, 55)

# make sub Img(white - img)
c, r = l.shape
outImg = np.ones((c,r), np.uint8)*255
#showImg('outImg', outImg)
#print(l.shape)
#print(outImg)
subImg = outImg - l
#showImg('subImg', subImg)

viewData = {'original':img, 'l':l, 'outImg':outImg, 'subImg':subImg}
viewResult2(viewData, 2, 'gray')

for i in range(1,6):
    subImg = outImg - subImg
showImg('subImg', subImg)

# make Reverse Image
reverseImg = rt.createReverseImg(filterImg)
#showImg('reverseImg', reverseImg)
reverseImg1 = rt.createReverseImg(labFilterImg)
grayReverseImg = cv2.cvtColor(reverseImg1, cv2.COLOR_LAB2BGR)
grayReverseImg = cv2.cvtColor(grayReverseImg, cv2.COLOR_BGR2GRAY)


titles = ['Filter Img - l', 'Filter Img - LAB', 'Reverse Img - l', 'Reverse Img - LAB', 'Gray Reverse Img - LAB']
images = [filterImg, labFilterImg, reverseImg, reverseImg1, grayReverseImg]
showPlot(titles, images)



# Image merge
mergeImg = rt.mergeImg(l, reverseImg)
mergeImg1 = rt.mergeImg(l, grayReverseImg)
BlendingImg = rt.imgBlending(l, reverseImg, 0.5)
BlendingImg1 = rt.imgBlending(l, grayReverseImg, 0.5)

titles = ["Merge Img - l", "Merge Img - LAB", "Blending Img - l", "Blending Img - LAB", 'Original Img']
images = [mergeImg, mergeImg1, BlendingImg, BlendingImg1, img]
showPlot(titles, images)

# viewData = {"Merge Img - l":mergeImg, "Merge Img - LAB":mergeImg1, "Blending Img - l":BlendingImg, "Blending Img - LAB":BlendingImg1, 'Original Img':img}
# viewResult2(viewData, 2)


resultImg = rt.globalThresholding(mergeImg)
resultImg1 = rt.globalThresholding(mergeImg1)
resultImg2 = rt.globalThresholding(BlendingImg)
resultImg3 = rt.globalThresholding(BlendingImg1)

# titles = ['Threshold - Merge_L', 'Threshold Merge_LAB', 'Threshold - Blending_L', 'Threshold - Blending_LAB', 'Original Img']
# images = [resultImg, resultImg1, resultImg2, resultImg3, img]
# showPlot(titles, images)

viewData = {'Threshold - Merge_L' : resultImg, 'Threshold Merge_LAB' : resultImg1, 'Threshold - Blending_L' : resultImg2, 'Threshold - Blending_LAB' : resultImg3, 'Original Img': img}
viewResult2(viewData, 2, 'gray')


# showImg('Result Img', resultImg)

# rt.thresholding(resultImg)
# rt.thresholding(resultImg1)
# rt.thresholding(resultImg2)
# rt.thresholding(resultImg3)


#viewData = {'Original': img, 'Gray Img' : img_gray, 'Alpha Img' : img_alp}








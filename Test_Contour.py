"""
    Test_Contour.py

        Code의 영역을 Contour로 따서 검정색으로 채움
        또는
        Canny를 수행하여...

        1. 코드의 Dot를 따기위해.. Contour의 최소값과 최대값을 알 수 있게 Test용 Pgm 제작
        2. 위에서 확인한 내용을 바탕으로.. 최소값과 최대값 사이의 Contour를 검정색으로 floodfill
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef
import os
import time




MIN, MAX, thrValue = 0,0,127
"""
defPath = 'img/'
#imgUrl = 'img/K-001.jpg'
imgUrl = 'img/K-003.jpg'
oimg = cv2.imread(imgUrl)
img = oimg.copy()
GrayImg = cv2.cvtColor(oimg, cv2.COLOR_BGR2GRAY)
cv2.imshow('original', img)
impDef.close_window()
"""



def onChange(x):

    print('onChange : Threshold = %d, Min = %d, Max = %d'%(thrValue, MIN, MAX))
    #img = reverse()

    #Contour Size Check

def erosion(img, kernelSize = 3, qty = 1):

    kernel = np.ones((kernelSize,kernelSize), np.uint8)
    # 3x3 크기의 1로 채워진 매트릭스를 생성. Erosion 및 Dilation 커널로 사용예정.

    erosion = cv2.erode(img, kernel, iterations = qty)

    return erosion


def dilation(img, kernelSize = 3, qty = 1):

    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    # 3x3 크기의 1로 채워진 매트릭스를 생성. Erosion 및 Dilation 커널로 사용예정.

    dilation = cv2.dilate(img, kernel, iterations = qty)

    return dilation


def dontWork(x):
    pass


def reverse(getImg):

    img = cv2.bitwise_not(getImg)


def viewResult(orginImg, resultImg, orginalTitle = "Original Img", resultTitle = "Result Img"):

    plt.subplot(121), plt.imshow(orginImg, cmap = 'gray')
    plt.title(orginalTitle), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(resultImg, cmap = 'gray')
    plt.title(resultTitle), plt.xticks([]), plt.yticks([])

    plt.show()




def saveFile(f, fileName, savePath = 'img/'):
    global defPath

    print('이미지를 저장하시려면 P Key를 누르세요!!')
    k = cv2.waitKey(0) & 0xFF
    if k == ord('p'):
        # 이미지 저장
        saveFileName = defPath+fileName
        saveFile = cv2.resize(f, dsize = (640, 480), interpolation = cv2.INTER_AREA)
        print('saveFileName = ', saveFileName)
        cv2.imwrite(saveFileName, saveFile)
        impDef.close_window()
    elif k == 27:
        impDef.close_window()



def Contrast(img):

    #cv2.imshow('original', img)
    hist_equal = cv2.equalizeHist(img)
    #cv2.imshow('equallizeHist', hist_equal)
    #impDef.close_window()

    """
    hist, bins = np.histogram(hist_equal, 256, [0,255])
    plt.fill_between(range(256), hist, 0)
    plt.xlabel('pixel value')
    #plt.show()

    res = np.hstack((img, hist_equal))
    plt.hist(img.ravel(), 256, [0,256])
    plt.hist(hist_equal.ravel( ), 256, [0, 256])
    #plt.show( )
    #cv2.imshow('hstack', res)
    """
    return hist_equal


def cvtGamma(img, gamma = 1):
    outImg = np.zeros(img.shape, img.dtype)
    rows, cols = img.shape

    #Create Table
    LUT = []

    for i in range(256):
        LUT.append(((i/255.0)**(1/gamma))*255)

    LUT = np.array(LUT, dtype = np.uint8)
    outImg = LUT[img]

    return  outImg


def addGaussian(img):
    gaussianImg = cv2.GaussianBlur(img, (5,5), 0)
    return gaussianImg


def addMedian(img, ksize):
    medianImg = cv2.medianBlur(img,ksize)
    return medianImg


def addBilateral(img):
    medianImg = cv2.bilateralFilter(img,5,75,75)
    return medianImg


def adaptiveGaussian(img, value = 55):
    GrayImg = img.copy()
    thr3 = cv2.adaptiveThreshold(GrayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, value, 2)
    #cv2.imshow('Adaptive Gaussian', thr3)
    #impDef.close_window()
    #cv2.imwrite('img\Adaptive_Gaussian.jpg', thr3)

    return thr3


def adaptiveMean(img, value = 55):
    GrayImg = img.copy()
    thr3 = cv2.adaptiveThreshold(GrayImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, value, 2)
    #cv2.imshow('Adaptive Mean', thr3)
    #impDef.close_window()
    #cv2.imwrite('img\Adaptive_Mean.jpg', thr3)

    return thr3


def addClahe(img):
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    img2 = clahe.apply(img)

    return img2


def addEqualizeHist(grayImg):
    equ = cv2.equalizeHist(img_grayScale)

    return equ

def testContour(getImg, savePath = 'img/'):
    global thrValue, MIN, MAX

    cv2.imshow('getImg', getImg)
    #GrayImg = cv2.cvtColor(getImg, cv2.COLOR_GRAY2RGB)
    #img = cv2.bitwise_not(getImg)
    #cv2.imshow('reverse', img)
    #impDef.close_window()

    print('testContour : thrValue = %s, MIM = %s, MAX = %s'%(thrValue, MIN, MAX))

    cv2.namedWindow('Contour_Min_Max')
    cv2.createTrackbar('Threshold', 'Contour_Min_Max', 0, 255, onChange)
    cv2.createTrackbar('Min', 'Contour_Min_Max',  0, 100, onChange)
    cv2.createTrackbar('Max', 'Contour_Min_Max', 0, 1000, onChange)

    cv2.setTrackbarPos('Min', 'Contour_Min_Max', MIN)
    cv2.setTrackbarPos('Max', 'Contour_Min_Max', MAX)
    cv2.setTrackbarPos('Threshold', 'Contour_Min_Max', thrValue)

    while True:
        thrValue = cv2.getTrackbarPos('Threshold', 'Contour_Min_Max')
        MIN = cv2.getTrackbarPos('Min', 'Contour_Min_Max')
        MAX = cv2.getTrackbarPos('Max', 'Contour_Min_Max')
        #viewImg = getImg.copy()
        viewImg = img.copy()
        #cv2.imshow('viewImg', viewImg)



        #threshold 2020.06.22 주석처리... 그레이 이미지를 받음으로 필요 엄을듯...
        ret, thr = cv2.threshold(viewImg, thrValue, 255, cv2.THRESH_BINARY_INV)
        #ret, thr = cv2.threshold(getImg, thrValue, 0, cv2.THRESH_BINARY_INV)
        #print('thrValue = %s '%thrValue)

        # contour 검출
        #_, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print('컨투어 갯수 = ', len(contours))

        # 일정 크기이상의 Contours만 검출
        for cnt in range(len(contours)):
            area = cv2.contourArea(contours[cnt])
            #print('area = %s, cnt = %s'%(area, cnt+1))
            
            #if area >= MIN and area <= MAX:
            if area <= MIN and area >= MAX:     # White 기준으로 동작...??? 검정색 검출시 반대로???
                #cv2.drawContours(viewImg, [contours[cnt]], 0, (0,0,255), 1)
                cv2.drawContours(thr, [contours[cnt]], 0, (0, 0, 255), -1)

        #print('cnt = %s'%cnt)
        #cv2.drawContours(img, contours, -1, (255,255,255), 1)
        #cv2.imshow('Contour_Min_Max', viewImg)
        cv2.imshow('Contour_Min_Max', thr)


        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'):
            # 이미지 저장
            saveFileName = savePath+"K_Contour_Img.jpg"
            saveFile = cv2.resize(thr, dsize = (640, 480), interpolation = cv2.INTER_AREA)
            print('saveFileName = ', saveFileName)
            cv2.imwrite(saveFileName, saveFile)

        elif k == 27:
            break
            cv2.destroyAllWindows()


def canny(img):

    cannyimg = img.copy()
    kernel = np.ones((3,3), np.uint8)
    cv2.namedWindow('canny')
    cv2.createTrackbar('low_Thr', 'canny', 0, 500, dontWork)
    cv2.createTrackbar('high_Thr', 'canny', 0, 500, dontWork)
    cv2.setTrackbarPos('low_thr','canny', 50)
    cv2.setTrackbarPos('high_thr', 'canny', 200)

    while True:
        low = cv2.getTrackbarPos('low_Thr', 'canny')
        high = cv2.getTrackbarPos('high_Thr', 'canny')

        imgCanny = cv2.Canny(cannyimg, low, high)
        #imgCanny = cv2.dilate(imgCanny, kernel, iterations =1)

        cv2.imshow('canny', imgCanny)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'):
            # 이미지 저장
            saveFileName = "img/K_Canny_Img.jpg"
            saveFile = cv2.resize(thr, dsize = (640, 480), interpolation = cv2.INTER_AREA)
            print('saveFileName = ', saveFileName)
            cv2.imwrite(saveFileName, saveFile)
            cv2.destroyAllWindows( )
        elif k == 27:
            break
            cv2.destroyAllWindows( )

    cv2.destroyAllWindows()





def lightEffect(img_BGR):

    # RGB To LAB : Image 변환(Color -> LAB), cv2.cvtColor(cv2.RGB2LAB)
    LabImg = cv2.cvtColor(img_BGR, cv2.COLOR_RGB2LAB)
    GrayImg = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

    #Median Filter
    filterImg = cv2.medianBlur(LabImg, 45)

    # Reverse Image
    filterImg = cv2.cvtColor(filterImg, cv2.COLOR_RGB2GRAY)
    revImg = cv2.bitwise_not(filterImg)
    #cv2.imshow('filterImg', filterImg)
    #cv2.imshow('revImg', revImg)


    # Composite

    # Add
    #finalImg = cv2.add(GrayImg, filterImg)
    #finalImg_rev = cv2.add(GrayImg, revImg)

    tempImg = cv2.cvtColor(LabImg, cv2.COLOR_LAB2BGR)
    tempImg = cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY)
    finalImg = cv2.add(tempImg, filterImg)
    finalImg_rev = cv2.add(tempImg, revImg)

    #viewResult(finalImg, finalImg_rev, "final Lab Img", "final Lab Img_rev")

    #Final Threshold
    ret, finThr = cv2.threshold(finalImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('final Threshold Otsu', finThr)

    #viewResult(oimg, finalImg, 'Final Image')
    #viewResult(finalImg, finalImg_rev, 'Final Image_rev')
    #viewResult(oimg, finThr, 'Final Threshold')

    return finalImg, finalImg_rev, finThr



"""
def imgReDraW(img, tileSize = 3, step = 1):

    #Image Gray Scale로 변환
    #gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = img.copy()

    #tileSize가 3보다 작은 경우 3
    if tileSize < 3: tileSize = 3

    #tileSize가 짝수인 경우 -1
    if tileSize%2: tileSize = tileSize-1


    #Image 정보 획득
    height, width = gimg.shape
    rtnImg = np.zeros((height, width), np.uint8)

    print('width = %s, height = %s'%(width, height))


    # tileSize의 평균밝기와 thr과 비교하여 각 픽셀(rtnImg(x,y))에 저장
    for y in range(0, height):
        # tile의 y축 시작값 과 끝값을 구한다...
        if y < (tileSize - 1) / 2:
            yStart = 0
        else:
            yStart = int(y - (tileSize - 1) / 2)

        yEnd = yStart + (tileSize - 1)
        if yEnd > height:
            yEnd = height
            yStart = yEnd - (tileSize-1)

        for x in range(0, width):
            #tile의 x축 시작값과 끝값을 구한다...
            if x < (tileSize-1)/2:
                xStart = 0
            else:
                xStart = int(x - (tileSize-1)/2)

            xEnd = xStart + (tileSize-1)

            if xEnd > width:
                xEnd = width
                xStart = xEnd - (tileSize-1)

            #tile의 평균값
            pxTot = 0
            for ty in range(yStart, yEnd, step):
                for tx in range(xStart, xEnd, step):
                    pxTot = pxTot+gimg.item(ty, tx)

            #tile의 평균값과 기준값 비교(thr)하여 resultThr 결정 밝으면 255, 어두우면 0
            pxAvg = int(pxTot/(tileSize*tileSize))
            if gimg.item(y,x) > pxAvg:
                rtnImg.itemset(y, x, 255)
            # rtnImg.itemset(y, x, resultThr)
    return rtnImg
"""



def imgReDraW2(img, tileSize = 3, step = 1):

    # imgReDraw2
    # 장점 : imgReDraw3 대비 약 0.5Sec ~ 1.5Sec 빠름
    # 단점 : TileSize/2 만큼은 변환하지 않음(가장자리 검정색으로 나타남)
    # 조건 ->  Gamma : 1.5, tileSize : 50, step : 10, Median : 3
    # 결과 : 1, 2, 3, 4, 6, 8, 9, 11, 13, 14, 15 성공


    #Image Gray Scale로 변환
    #gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = img.copy()

    #tileSize가 3보다 작은 경우 3
    if tileSize < 3: tileSize = 3

    #tileSize가 짝수인 경우 -1
    if tileSize%2 : pass
    else : tileSize = tileSize-1

    # tileSize보다 Step이 큰 경우
    if tileSize < step: step = tileSize

    gain = int((tileSize-1)/2)  # Tile Size

    #Image 정보 획득
    height, width = gimg.shape
    rtnImg = np.zeros((height, width), np.uint8)

    print('width = %s, height = %s'%(width, height))


    yStartPx = gain             # Y축 시작 점
    yEndPx = height - gain      # Y축 끝나는 지점
    xStartPx = gain             # X축 시작 점
    xEndPx = width - gain       # X축 끝나는 점

    # Debuging용 File생성
    #f = open("Debuging.txt", 'w', encoding = "UTF8")

    #f.write("width / height / yStartPx / yEndPx / xStartPx / xEndPx / yTileStart / yTileEnd / xTileStart / xTileEnd / ty / tx / pxTot \n")
    # tileSize의 평균밝기와 thr과 비교하여 각 픽셀(rtnImg(x,y))에 저장
    for y in range(yStartPx, yEndPx):

        yTileStart = y-gain
        yTileEnd = y+gain
        for x in range(xStartPx, xEndPx):
            xTileStart = x-gain
            xTileEnd = x+gain
            #tile의 평균값
            pxTot = 0
            stepCnt = 0
            pxValue = gimg.item(y,x)
            for ty in range(yTileStart, yTileEnd+1, step):
                for tx in range(xTileStart, xTileEnd+1, step):
                    pxTot = pxTot+gimg.item(ty, tx)
                    stepCnt = stepCnt+1
                    #fData = "w = %s, h = %s, ySP = %s, yEP = %s, xSP=%s, xEP = %s, yTS = %s, yTE = %s, xTS = %s, xTE = %s, ty = %s, tx = %s, gimg.item(%s, %s) = %s, pxTot = %s \n"%\
                        #(width, height, yStartPx, yEndPx, xStartPx, xEndPx, yTileStart, yTileEnd, xTileStart, xTileEnd, ty, tx, y, x, gimg.item(y,x) , pxTot)
                    #f.write(fData)
            #tile의 평균값과 기준값 비교(thr)하여 resultThr 결정 밝으면 255, 어두우면 0
            #pxAvg = int((pxTot-pxValue)/(tileSize*tileSize-1))
            pxAvg = int((pxTot - pxValue) / (stepCnt - 1))
            if gimg.item(y,x) >= pxAvg: rtnImg.itemset(y, x, 255)
            #elif gimg.item(y,x) == pxAvg : rtnImg.itemset(y, x, 127)
            #print('x = %s, y = %s'%(x,y))
            #fData = "x = %s, y = %s, pxAvg = %s, gimg.item(%s, %s) = %s \n"%(x, y, pxAvg, y, x, gimg.item(y,x))
            #f.write(fData)
            # rtnImg.itemset(y, x, resultThr)
    #f.close()
    return rtnImg



def imgReDraW3(img, tileSize = 3, step = 1):

    # imgReDraw3
    # 단점 : imgReDraw2 대비 약 0.5Sec ~ 1Sec 느림
    # 장점 : 가장자리 전체를 다 변환할 수 있음
    # 조건 ->  Gamma : 1.5, tileSize : 50, step : 10, Median : 3
    # 결과 :  1, 2, 3, 4, 6, 8, 9, 11, 13, 14, 15 성공

    #Image Gray Scale로 변환
    #gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = img.copy()

    #tileSize가 3보다 작은 경우 3
    if tileSize < 3: tileSize = 3

    #tileSize가 짝수인 경우 -1
    if tileSize%2 : pass
    else : tileSize = tileSize-1

    # tileSize보다 Step이 큰 경우
    if tileSize < step: step = tileSize

    gain = int((tileSize-1)/2)  # Tile Size

    #Image 정보 획득
    height, width = gimg.shape
    rtnImg = np.zeros((height, width), np.uint8)

    print('width = %s, height = %s'%(width, height))

    for y in range(0, height):
        # tile의 y축 시작값 과 끝값을 구한다...
        if y < (tileSize - 1) / 2:
            yStart = 0
        else:
            yStart = int(y - (tileSize - 1) / 2)

        yEnd = yStart + (tileSize - 1)
        if yEnd > height:
            yEnd = height
            yStart = yEnd - (tileSize - 1)

        for x in range(0, width):
            # tile의 x축 시작값과 끝값을 구한다...
            if x < (tileSize - 1) / 2:
                xStart = 0
            else:
                xStart = int(x - (tileSize - 1) / 2)

            xEnd = xStart + (tileSize - 1)

            if xEnd > width:
                xEnd = width
                xStart = xEnd - (tileSize - 1)

            # tile의 평균값
            pxTot = 0
            pxValue = gimg.item(y, x)
            stepCnt = 0
            for ty in range(yStart, yEnd, step):
                for tx in range(xStart, xEnd, step):
                    pxTot = pxTot + gimg.item(ty, tx)
                    stepCnt = stepCnt + 1
            # tile의 평균값과 기준값 비교(thr)하여 resultThr 결정 밝으면 255, 어두우면 0
            pxAvg = int((pxTot - pxValue) / (stepCnt - 1))
            if gimg.item(y, x) >= pxAvg: rtnImg.itemset(y, x, 255)


    return rtnImg



def reDrawImg(grayImg, tileSize, step, median, erosionKernelSize, dilationKernelSize):
# def reDrawImg(grayImg, tileSize, step, median, eQty, dQty):
    #sTime = time.time()

    # adaptiveMean
    aImg = adaptiveMean(grayImg)

    # erosion
    eImg = erosion(aImg, erosionKernelSize, 1)

    # dilation
    dImg = dilation(eImg, dilationKernelSize, 1)

    reDrawImg = imgReDraW2(dImg, tileSize, step)
    reDraw_M = addMedian(reDrawImg, median)
    reDraw_E = erosion(reDraw_M)
    reDraw_D = dilation(reDraw_M)

    #print(time.time() - sTime)

    rtnImg = [aImg, reDrawImg, reDraw_M, reDraw_E, reDraw_D]

    return rtnImg


########################################################################################################################
"""
#canny()
#imgName = imgUrl + "_Canny.jpg"
#oimg = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
#img = oimg.copy()
testContour(img)
#canny()

testImg = img.copy()


gaussianImg = addGaussian(testImg)
cv2.imshow('addGaussian', gaussianImg)
contrastImg = Contrast(gaussianImg)
cv2.imshow('gaussian + contrastImg', contrastImg)
canny(contrastImg)
testContour(contrastImg)
impDef.close_window()


medianImg = addMedian(testImg, 3)
cv2.imshow('addMedian', medianImg)
contrastImg = Contrast(medianImg)
cv2.imshow('median + contrastImg', contrastImg)
canny(contrastImg)
testContour(contrastImg)
impDef.close_window()


bilateralImg = addBilateral(testImg)
cv2.imshow('bilateralImg', bilateralImg)
contrastImg = Contrast(bilateralImg)
cv2.imshow('bilateral + contrastImg', contrastImg)
#canny(contrastImg)
testContour(contrastImg)
impDef.close_window()

#canny(contrastImg)
#testImg = cv2.imread('img/K-003.jpg_Canny.jpg', cv2.IMREAD_GRAYSCALE)
#testContour(gaussianImg)



aGaussianImg = adaptiveGaussian(testImg, value = 55)
viewResult(img, aGaussianImg, 'Add Adaptive Gaussian Image')
#cv2.imshow('adaptiveGaussian', aGaussianImg)
aMeanImg = adaptiveMean(testImg, value = 55)
viewResult(img, aMeanImg, 'Add Adaptive Mean Image')
#cv2.imshow('adaptiveMean', aMeanImg)
#impDef.close_window()

#adaptive Gaussina + contour
#testContour(aGaussianImg)
canny(aGaussianImg)
#adaptive Mean + contour
#testContour(aMeanImg)
"""
"""
lightEffectImg, lightEffectImg_rev, lightEffectImg_Thr  = lightEffect(oimg)


cv2.imshow('lightEffectImg', lightEffectImg)
saveFile(lightEffectImg, "k_LightEffectImg.jpg")

cv2.imshow('lightEffectImg_rev', lightEffectImg_rev)
saveFile(lightEffectImg_rev, "k_LightEffectImg_rev.jpg")

cv2.imshow('lightEffectImg_Thr', lightEffectImg_Thr)
saveFile(lightEffectImg_Thr, "k_LightEffectImg_Thr.jpg")


testContour(lightEffectImg_rev)
canny(lightEffectImg_rev)
ClaheImg = addClahe(lightEffectImg_rev)

dst = np.hstack((lightEffectImg_rev, ClaheImg))
cv2.imshow('Clahe Img', dst)
impDef.close_window()

testContour(dst)
"""






"""
########################################################################################################################
#2020.06.17 - reDraw... Test
# 1. Gray Image 생성
# 2. Pixel값을 기준값(thr)과 비교하여 Pixel 값이 크면 255, 작으면 0으로 변환(???)
# 3. np배열에 2번에서 변환한 값으로 저장

sampleImg = ['TestImg/1.jpg', 'TestImg/2.jpg','TestImg/3.jpg', 'TestImg/4.jpg', 'TestImg/5.jpg',
             'TestImg/6.jpg', 'TestImg/7.jpg', 'TestImg/8.jpg', 'TestImg/9.jpg', 'TestImg/10.jpg',
             'TestImg/11.jpg', 'TestImg/12.jpg', 'TestImg/13.jpg', 'TestImg/14.jpg', 'TestImg/15.jpg', 'TestImg/DW_16.jpg']

#sampleImg = ['TestImg1/1.jpg', 'TestImg1/2.jpg']
#sampleImg = ['TestImg/4.jpg']

cnt = 1
tileSize = 50
step = 10
median = 3

path = 'TestImg/'

for i in sampleImg:
    img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    imgTitle = 'GrayScale - '+i
    img1 = cvtGamma(img, 1.5)
    #print(img.item(100,100))
    #img.itemset(100,100,127)
    #cv2.imshow(imgTitle, img1)
    #viewResult(img, img1, "original", "Gamma 2")
    #impDef.close_window()
    #testContour(img1, 'TestImg/')
    #canny(img1)

    # imgReDraw(gray image, Tile Size, Pixel 단계)
    # gray image
    # tile Size : 기준 Pixel과 비교할 주위의 Pixel 크기 예) 3x3 또는 5x5
    # Pixel 단계 : VGA 기준 640 x 480 만큼 Pixel 검사를 해야함.. 넘 오래걸려서.. 2입력시 320 x 240만 진행(2의 배수, 3의 배수 등)


 
    start = time.time()
    reDrawImg = imgReDraW(img, tileSize)
    viewTxt = "ReDraw Image - "+i
    #viewResult(img, reDrawImg, "original", viewTxt)
    print(time.time()-start)

    #cv2.imshow(viewTxt, reDrawImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #testContour(img, "TestImg1/")


    start = time.time( )
    img1 = cvtGamma(img, 1.5)
 ###########################################################
    # imgReDraw2
    # 장점 : imgReDraw3 대비 약 0.5Sec ~ 1.5Sec 빠름
    # 단점 : TileSize/2 만큼은 변환하지 않음(가장자리 검정색으로 나타남)
    # 조건 ->  Gamma : 1.5, tileSize : 50, step : 10, Median : 3
    # 결과 : 1, 2, 3, 4, 6, 8, 9, 11, 13, 14, 15 성공

    #reDrawImg = imgReDraW2(img1, tileSize, step)


###########################################################
    # imgReDraw3
    # 단점 : imgReDraw2 대비 약 0.5Sec ~ 1Sec 느림
    # 장점 : 가장자리 전체를 다 변환할 수 있음
    # 조건 ->  Gamma : 1.5, tileSize : 50, step : 10, Median : 3
    # 결과 :  1, 2, 3, 4, 6, 8, 9, 11, 13, 14, 15 성공
    
    #reDrawImg = imgReDraW3(img1, tileSize, step)

###############################################################

    print(time.time( ) - start)
    #cv2.imshow('reDrawImg', reDrawImg)
    #impDef.close_window()

    #testContour(reDrawImg, 'TestImg/')
    reDrawImgMedian = addMedian(reDrawImg, median)
    erosionImg = erosion(reDrawImgMedian)
    dilationImg = dilation(reDrawImgMedian)

    #viewTxt = "ReDraw Image - " + i
    #fileName0 = "TestImg1/reDrawImg_"+str(cnt) + "_T" + str(tileSize)+"_S" + str(step)+".jpg"
    #fileName = "TestImg1/reDrawImg_"+str(cnt)+"_T"+str(tileSize)+"_S" + str(step)+"_M"+str(median)+".jpg"
    #fileName2 = "TestImg1/reDrawImg_"+str(cnt)+"_T"+str(tileSize)+"_S" + str(step)+"_M"+str(median)+"_E.jpg"
    #fileName3 = "TestImg1/reDrawImg_"+str(cnt)+"_T"+str(tileSize)+"_S" + str(step)+"_M"+str(median)+"_D.jpg"

    fileName0 = path+"reDraw/reDraw_" + str(cnt) + "_T" + str(tileSize) + "_S" + str(step) + ".jpg"
    fileName =  path+"reDraw/reDraw_"  + str(cnt) + "_T" + str(tileSize) + "_S" + str(step) + "_M" + str(median) + ".jpg"
    fileName2 = path+"reDraw/reDraw_"  + str(cnt) + "_T" + str(tileSize) + "_S" + str(step) + "_M" + str(
        median) + "_E.jpg"
    fileName3 = path+"reDraw/reDraw_"  + str(cnt) + "_T" + str(tileSize) + "_S" + str(step) + "_M" + str(
        median) + "_D.jpg"

   # canny(reDrawImgMedian)
   # path = "TestImg1/"
   # testContour(reDrawImgMedian, path)

    cv2.imwrite(fileName0, reDrawImg)
    print(fileName0)
    cv2.imwrite(fileName, reDrawImgMedian)
    #viewResult(img1,reDrawImg, 'Original', 'ReDraw' )


    print(fileName)
    imgLoad = cv2.imread(fileName)
    #testContour(imgLoad, "TestImg1/")
    #canny(imgLoad)

    cv2.imwrite(fileName2, erosionImg)
    print(fileName2)
    cv2.imwrite(fileName3, dilationImg)
    print(fileName3)
    #viewResult(erosionImg, dilationImg, 'Add Erosion', 'Add Dilation')

    cnt = cnt + 1


#imgLoad1 = cv2.imread('TestImg1/reDrawImg_1Tile15_Median5.jpg')
#testContour(imgLoad1, 'TestImg1/')




    #cv2.imshow(viewTxt, reDrawImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows( )

    saveFileName = "TestImg/Result_img"+"Tile"+str(tileSize)+".jpg"
    saveFileName1 = "TestImg/Result2_img"+"Tile"+str(tileSize)+".jpg"
    img2 = addMedian(reDrawImg2, median)
    saveFileName2 = "TestImg/Result2_img" + "Tile" + str(tileSize) + "_Medial"+str(median)+".jpg"
    cv2.imwrite(saveFileName, reDrawImg)
    cv2.imwrite(saveFileName1, reDrawImg2)
    print(saveFileName)
    print(saveFileName1)
    print(saveFileName2)
    cv2.imwrite(saveFileName2, img2)
    cv2.imshow('original Image', img)
    cv2.waitKey(0)
    viewResult(reDrawImg, reDrawImg2, "reDraw", "reDraw 2")
    viewResult(reDrawImg2, img2, "reDraw 2", "reDraw 2 + Median")
  
    #impDef.close_window()
"""
#2020-06-23 finish Test

################################################################################################################
#
#       Gradation Img Test
#       2020-06-23....
#

"""
inPath = 'Gradation_Img/'
outPath ='ReDrawImg/'
tileSize = 50
step = 10
median = 3
i = 1
fileList = os.listdir(inPath)
for file in fileList:
    imgName = inPath + file
    print(imgName)
    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)

    #_, _, lImg = lightEffect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    #saveFileName_L = outPath + 'reDraw' + str(i) + '_L.jpg'
    #cv2.imwrite(saveFileName_L, lImg)
    #print(saveFileName_L)

    # adaptiveMean
    aImg = adaptiveMean(img)
    saveFileName_A = outPath + 'reDraw' + str(i) + '_A.jpg'
    cv2.imwrite(saveFileName_A, aImg)
    print(saveFileName_A)

    # adaptiveGaussian
    #aImg = adaptiveGaussian(img)
    #saveFileName_A = outPath + 'reDraw' + str(i) + '_A.jpg'
    #cv2.imwrite(saveFileName_A, aImg)
    #print(saveFileName_A)


    sImg = erosion(aImg)
    sImg = dilation(sImg)
    sImg = dilation(sImg)


    #dilationImg = dilation(img)
    #saveFileName_D = outPath + 'reDraw' + str(i) + '_D.jpg'
    #cv2.imwrite(saveFileName_D, dilationImg)
    #print(saveFileName_D)


    #gammaImg = cvtGamma(img, 1.5)
    #viewResult(img, gammaImg, 'Original', 'Gamma 1.5')

    # ReDrawImg
    start = time.time( )
    #reDrawImg = imgReDraW2(gammaImg, tileSize, step)
    reDrawImg = imgReDraW2(sImg, tileSize, step)
    print(time.time( ) - start)

    saveFileName = outPath+'redraw'+str(i)+'.jpg'
    cv2.imwrite(saveFileName, reDrawImg)
    print(saveFileName)

    #노이즈 제거를 위한 Median Filter
    reDraw_M = addMedian(reDrawImg, median)
    saveFileName_M = outPath+'reDraw'+str(i)+'_M.jpg'
    cv2.imwrite(saveFileName_M, reDraw_M)
    print(saveFileName_M)


    #Erosion
    erosionImg = erosion(reDraw_M)
    saveFileName_E = outPath + 'reDraw' + str(i) + '_E.jpg'
    cv2.imwrite(saveFileName_E, erosionImg)
    print(saveFileName_E)

    #Dilation
    dilationImg = dilation(reDraw_M)
    saveFileName_D = outPath + 'reDraw' + str(i) + '_D.jpg'
    cv2.imwrite(saveFileName_D, dilationImg)
    print(saveFileName_D)

    i = i + 1
"""

######################################################################################
#
#   현재 최적
#   tileSize = 50, step = 10, median = 3
#   gray Image -> Adaptive Mean 작업
#   adaptive Mean Image에 erosion 진행
#   위 영상에 dilation 2회 진행
#   imgRedraW2 진행
#   리턴이미지

inPath = 'Gradation_Img/'
outPath ='ReDrawImg/'
tileSize = 50
step = 10
median = 3
i = 1

fileList = os.listdir(inPath)
for file in fileList:
    imgName = inPath + file
    print(imgName)
    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)

    #rtnImg = reDrawImg(img, tileSize, step, median, 3, 3)   # 약 2.6초 소요, 하지만 MCR Decoding 속도가 느림
    #rtnImg = reDrawImg(img, tileSize, step, median, 3, 5)  # 약 2.7초 소요(성공 : 20, 실패 : 20)
    #rtnImg = reDrawImg(img, 40, 5, median, 3, 5)  # 약 5.4초 소요(성공 : 20, 실패 : 20)
    #rtnImg = reDrawImg(img, tileSize, step, median, 3, 9)  # 약 2.7초 소요(거의 인식불가)

    #aImgName = outPath+"reDraw"+str(i)+"_AtM.jpg"
    #reDImgName = outPath + "reDraw" + str(i)+".jpg"
    #reDMName = outPath + "reDraw" + str(i)+"_M.jpg"
    #reDEName = outPath + "reDraw" + str(i) + "_E.jpg"
    #reDDName = outPath + "reDraw" + str(i) + "_D.jpg"

    #cv2.imwrite(aImgName, rtnImg[0])
    #print(aImgName)
    #cv2.imwrite(reDImgName, rtnImg[1])
    #print(reDImgName)
    #cv2.imwrite(reDMName, rtnImg[2])
    #print(reDMName)
    #cv2.imwrite(reDEName, rtnImg[3])
    #print(reDEName)
    #cv2.imwrite(reDDName, rtnImg[4])
    #print(reDDName)

    sTime = time.time()

    # adaptivMean : 55 & Erosion : 3 & Dilation : 5 -> 약 0.001초 소요(성공 : 18 , 실패 : 22)
    #rtnImg = adaptiveMean(img)
    #erosionImg = erosion(rtnImg)
    #dilationImg = dilation(erosionImg, 5)
    #rtnImgName = outPath+str(i)+"_AdM55"+".jpg"
    #erosionImgName = outPath+str(i)+"_AdM55_E3"+".jpg"
    #dilationImgName = outPath + str(i) + "_AdM55_E3_D3" + ".jpg"
    #print(time.time( ) - sTime)
    #cv2.imwrite(rtnImgName, rtnImg)
    #cv2.imwrite(erosionImgName, erosionImg)
    #cv2.imwrite(dilationImgName, dilationImg)


    # adaptivMean : 75 & Erosion : 3 & Dilation : 5 -> 약 0.001초 소요(성공 : 23 , 실패 : 17)
    # adaptivMean : 75 & Erosion : 3 & Dilation : 3~5 -> 약 0.002초 이내(성공 : 23  , 실패 : 17 )
    # adaptivMean : 75 & Erosion : 3 & Dilation : 4~7 -> 약 0.002초 이내(성공 :  28 , 실패 : 12 )
    rtnImg = adaptiveMean(img, 75)
    erosionImg = erosion(rtnImg)

    #dilationImg3 = dilation(erosionImg, 3)
    dilationImg4 = dilation(erosionImg, 4)
    dilationImg5 = dilation(erosionImg, 5)
    dilationImg6 = dilation(erosionImg, 6)
    dilationImg7 = dilation(erosionImg, 7)


    rtnImgName = outPath+str(i)+"_AdM75"+".jpg"
    erosionImgName = outPath+str(i)+"_AdM75_E3.jpg"
    #dilationImgName3 = outPath + str(i) + "_AdM75_E3_D3" + ".jpg"
    dilationImgName4 = outPath + str(i) + "_AdM75_E3_D4" + ".jpg"
    dilationImgName5 = outPath + str(i) + "_AdM75_E3_D5" + ".jpg"
    dilationImgName6 = outPath + str(i) + "_AdM75_E3_D6" + ".jpg"
    dilationImgName7 = outPath + str(i) + "_AdM75_E3_D7" + ".jpg"
    print(time.time( ) - sTime)

    cv2.imwrite(rtnImgName, rtnImg)
    cv2.imwrite(erosionImgName, erosionImg)
    #cv2.imwrite(dilationImgName3, dilationImg3)
    cv2.imwrite(dilationImgName4, dilationImg4)
    cv2.imwrite(dilationImgName5, dilationImg5)
    cv2.imwrite(dilationImgName6, dilationImg6)
    cv2.imwrite(dilationImgName7, dilationImg7)

    i = i + 1














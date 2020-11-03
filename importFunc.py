"""
    importFunc.py

        threshold(threshold, value)
        erosion(img, kernelSize = 3, qty = 1)
        dilation(img, kernelSize = 3, qty = 1)
        reverse(getImg)
        viewResult(orginImg, resultImg, orginalTitle = "Original Img", resultTitle = "Result Img")
        imgConcat(img1, img2, type = 1 or 0)    영상 붙이기 1 : 가로, 0 : 세로
        imgStack(img1, img2, type = 1 or 0)     영상 붙이기 1 : 가로, 0 : 세로
        saveFile(f, fileName, savePath = 'img/')
        cvtGamma(img, gamma = 1)

        addGaussian(img)
        addMedian(img, ksize)
        addBilateral(img)
        adaptiveGaussian(img, value = 55)
        adaptiveMean(img, value = 55)
        addClahe(img)
        addEqualizeHist(grayImg)

        testContour(getImg, savePath = 'img/')
        canny(img)
        lightEffect(img_BGR)
        imgReDraW3(img, tileSize = 3, step = 1)
        reDrawImg(grayImg, tileSize, step, median, erosionKernelSize, dilationKernelSize)
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import default_import as impDef
import os
import time


def threshold(threshold, value):
    global img, GrayImg, LabImg
    ret, thr = cv2.threshold(img, threshold, value, cv2.THRESH_BINARY)
    #cv2.imshow('origina - Threshold', thr)
    #impDef.close_window()

    return thr


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


def reverse(getImg):

    img = cv2.bitwise_not(getImg)

    return img


def viewResult(orginImg, resultImg, orginalTitle = "Original Img", resultTitle = "Result Img"):

    plt.subplot(121), plt.imshow(orginImg, cmap = 'gray')
    plt.title(orginalTitle), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(resultImg, cmap = 'gray')
    plt.title(resultTitle), plt.xticks([]), plt.yticks([])

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

def imgConcat(img1, img2, type = 1):
    if type:
        rtnimg = cv2.hconcat([img1, img2])
    else:
        rtnimg = cv2.vconcat([img1, img2])

    return rtnimg


def imgStack(img1, img2, type = 1):
    if type:
        rtnimg = np.hstack((img1, img2))
    else:
        rtnimg = np.vstack((img1, img2))

    return rtnimg


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






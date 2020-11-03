import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def showImage():
    imgfile = 'img/100.jpg'

    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(imgfile, 0) # GrayScale
    img_alp = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)

    cv2.imshow('Kyung Hyun', img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    cv2.imshow('Kyung Hyun - GrayScale', img_gray)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()

    cv2.imshow('Kyung Hyun - Unchanged', img_alp)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    # Window Size 변경 가능
    cv2.namedWindow('Kyung Hyun - NORMAL', cv2.WINDOW_NORMAL)
    cv2.imshow('Kyung Hyun - NORMAL', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Image 복사
    k = cv2.waitKey(0) & 0xFF

    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('c'):
        cv2.imwrite("img/100_copy.jpg", img)
        cv2.destroyAllWindows()

    #Matplotlib 사용하기(비율맞춰 Size변경, 저장 등 네비게이션)
    #img_gray = cv2.imread(imgfile, 0)  # GrayScale
    #plt.imshow(img_gray, cmap = 'gray', interpolation='bicubic')
    #plt.xticks([])
    #plt.yticks([])
    #plt.title('Kyung Hyun')
    #plt.show()




def viewResult2(viewData, row):
#viewData = {'Original': img, 'Gray Img' : img_gray, 'Alpha Img' : img_alp}

    dataLen = len(viewData)
    cols = math.ceil(dataLen/row)
    print("cols = ", cols)

    i = 1
    for key, val in viewData.items():
        subplotNo = str(cols)+str(row)+str(i)
        print('key = ', key)
        print('subplotNo = ', subplotNo)
        #cv2.imshow(key, val)

        #plt.subplot(subplotNo), plt.imshow(val, cmap = 'gray')
        plt.subplot(subplotNo), plt.imshow(val)
        plt.title(key), plt.xticks([]), plt.yticks([])

        i = i + 1
    # End of for key, val int viewData.items():

    plt.show()
# End of viewResult2()





imgfile = 'img/100.jpg'
img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
img_gray = cv2.imread(imgfile, 0) # GrayScale
img_alp = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
img_4 = img

viewData = {'Original': img, 'Gray Img' : img_gray, 'Alpha Img' : img_alp}
LEN = len(viewData)
print('LEN = ', LEN)

row = 3
viewResult2(viewData, row)


'''
plt.subplot(334), plt.imshow(img_gray)
plt.xticks([]), plt.yticks([])

plt.subplot(332), plt.imshow(img)
plt.xticks([]), plt.yticks([])

plt.subplot(336), plt.imshow(img_alp)
plt.xticks([]), plt.yticks([])

plt.subplot(338), plt.imshow(img_4)
plt.xticks([]), plt.yticks([])
'''
plt.subplot(131), plt.imshow(img_gray)
plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(img)
plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(img_alp)
plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(img_4)
plt.xticks([]), plt.yticks([])

plt.show()


#plt.title(key), plt.xticks([]), plt.yticks([])


#showImage()
cv2.destroyAllWindows()





#viewData = {'Original': img, 'Gray Img' : img_gray, 'Alpha Img' : img_alp}
#LEN = len(viewData)
#print('LEN = ', LEN)


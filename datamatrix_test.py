from pylibdmtx.pylibdmtx import decode
from cv2 import cv2

try:
    img = cv2.imread('img/datamatrix.jpg')
except:
    print('Camera Loading Fail!!')

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#ret, thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#thr = img

#decode(cv2.imread('img/datamatrix.jpg'))
#decode(cv2.imread('datamatrix.jpg'))
img = cv2.imread('datamatrix.jpg')
thr = img
msg = pylibdmtx.decode(thr, timeout = 3000)
msg = "Can't Decode!"
if msg:
    mcrResult = "MCR Result : {0}".format(msg)
else:
    msg = "This Image Can't Decode!"

print(mcrResult)

if cv2.waitKey(1) & 0xFF == ord('q'):
    cap.release( )
    cv2.destroyAllWindows( )




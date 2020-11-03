from pylibdmtx.pylibdmtx import decode
from cv2 import cv2
import os
import time



class DecodingMCR:    
        
    def __init__(self, img, timeout, maxcount):
        self.img = img
        self.timeout = timeout
        self.maxcount = maxcount
   
    def Decoding(self):   
        #
        # 
        #h, w  = img.shape[:2]
        #result = decode((img[:, :, :1].tobytes(), w, h), timeout = self.timeout)    
        
        result = decode(self.img, timeout = self.timeout, max_count = self.maxcount)
        if result:  # Success Decoding
            decode_object = result[0]
            print("MCR Result : {0}".format(result))
            msg = ("Decoded.data = {0}".format(decode_object.data))
            return msg

        else :      # Fail Decoding
            msg = ("Decoding Fail !")
            return msg

# Setting
timeout = 3000  # 3초
maxcount = 1

# Setting path
inPath = "C:/Python_Test/Test_Image/"
fileList = os.listdir(inPath)

for file in fileList:
    imgName = file
    imgFile = inPath+file
    print("imgFile = {0}".format(imgFile))
    
    # Test
    img = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(imgFile)


    decodeImg = DecodingMCR(img, timeout, maxcount)
    stime = time.time()
    msg = decodeImg.Decoding()
    tactTime = round(time.time() - stime, 2)
    print("Decoding... -> {0} : Tact Time : {1}".format(msg, tactTime))

    # cv2.putText(img, msg, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # cv2.imshow(imgName, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows( )






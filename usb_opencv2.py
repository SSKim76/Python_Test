from cv2 import cv2

cap = cv2.VideoCapture(0)

width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

cap.set(3, width)
cap.set(4, height)

input_txt = "width:"+width+", height:"+height+", fps:"+fps
input_font = "cv2.FONT_HERSHEY_SCRIPT_SIMPLEX"
input_font_size = 1
input_font_color = "255, 255, 255"
input_font_pos = "10, 50"

while(True):
    ret, frame = cap.read()
    cv2.putText(frame, input_txt, (input_font_pos), input_font, input_font_size, (input_font_color))
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

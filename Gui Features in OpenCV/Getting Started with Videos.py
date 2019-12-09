import numpy as np
import cv2

FPS = 30
resize_frame = (320, 240)
resized = True

cap = cv2.VideoCapture('F:\\Code\\XVideoEdit\\XVideoEdit\\chapter7.mp4')

# Define a codec and create VideoWriter
four_cc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
if resized:
    frame_size = resize_frame
else:
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # must be int()
out = cv2.VideoWriter('output.avi', four_cc, FPS, frame_size)

# framesize要与原视频保持一致，否则写入不成功; 或者自定义framesize要将原视频resize后写入

while True:
    ret, frame = cap.read()
    if ret:
        if resized:
            frame = cv2.resize(frame, frame_size)
        out.write(frame)
        cv2.imshow('play', frame)
        if cv2.waitKey(int(1000 / FPS)) & 0xff == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()

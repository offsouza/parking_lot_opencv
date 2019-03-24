import cv2  
import imutils
from imutils import perspective
import numpy as np
from matplotlib import pyplot as plt 
import time
vs = cv2.VideoCapture('p.mp4')
fps = 12
#capSize = (640, 480) 
#capSize = (1280,720)
capSize = (640,360)
#capSize = (1920,1080)
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter()
success = out.open('./teste5.mp4',fourcc,fps,capSize,True)
check = False
while vs.isOpened():
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=700)   


    minX, minY, maxX, maxY = 287, 55, 366, 170
    box = frame[minY:maxY, minX:maxX]

    box = imutils.resize(box, width=200)

    


    gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray ", gray )
    print(gray.shape)
   
    

    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    cv2.imshow("gau ", gray )

    #gray = cv2.medianBlur(gray,17)
    #cv2.imshow("media ", gray )

    light = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("light", light)

    #thresh = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("Otsu", thresh)

    
    thresh = cv2.dilate(light, None, iterations=10)
    thresh = cv2.erode(thresh, None, iterations=10)

    thresh = cv2.erode(thresh, None, iterations=10)
    thresh = cv2.dilate(thresh, None, iterations=20)


    cv2.imshow("thresh", thresh)

    #box = np.array([[minX, minY], [maxX, maxY]])
    #rect = cv2.minAreaRect(box)
    #box = np.int0(cv2.cv.BoxPoints(rect)) if imutils.is_cv2() else cv2.boxPoints(rect)

    #print(box)
    #plate = perspective.four_point_transform(frame, box)

    cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)


    #cv2.imshow("Frame", box)
    #cv2.imshow("Perspective Transform", gray)

    out.write(frame)
    key = cv2.waitKey(1) & 0xFF

    check = False

    if check == False:
        frame1 = np.array(thresh)
        a = np.count_nonzero(frame1==0)
        print('A: %s' %a)
        soma = sum(thresh)
        soma = sum(soma)
        #print(soma)
        check = True

    if soma < 10000:
        cv2.putText(frame, 'OK', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.35, (0, 0, 255), 3)

    cv2.imshow("frame", frame)
    # Se a tecla 'q' for pressionada, finaliza com o loop
    if key == ord("q"):
        break

    if key == ord('p'):
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        plt.plot(hist)
        plt.show()

    time.sleep(0.1)
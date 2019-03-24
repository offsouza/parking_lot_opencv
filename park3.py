import cv2  
import imutils
from imutils import perspective 
from imutils.video import count_frames
import numpy as np
from matplotlib import pyplot as plt 
import time
from ferramentas import image_utils
from svm import SVM


path = 'videos/p2.mp4'
vs = cv2.VideoCapture(path)
fps = 12
capSize = (640,360)
#capSize = (1920,1080)
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter()
success = out.open('./teste5.mp4',fourcc,fps,capSize,True)

num_frames = count_frames(path)
print(num_frames)


check = True
i = 0 

while i < num_frames:
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=450)  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray ", gray )

    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imshow("gau ", gray )
    

    '''if check == True:
        boxes = getSpotsCoordiantesFromImage(frame, 2)
        print(boxes)
        check = False'''

    box1 = np.array([(155.5483870967742, 190.4596774193548), (206.35483870967744, 194.08870967741933), (196.375, 227.6572580645161), (139.21774193548384, 226.74999999999994)])
    box2 = np.array ([(131.05241935483872, 227.6572580645161), (82.06048387096773, 218.58467741935482), (112.90725806451611, 187.7379032258064), (151.9193548387097, 190.4596774193548)])
    boxes = [box1, box2]
    #print('afdasdf')
    #print(box1[0].shape)

    img_resize = image_utils.getRotateRect(gray, boxes)
    feature = image_utils().extract_features(img_resize)

    '''feature1 = feature.reshape(-1, 1)
    score0 = SVM().predict(feature1)
    score1 = SVM().predict(feature[1])

    print (score0, score1)'''

    score = SVM().predict(feature)

    if score[0] == 0: 
        cv2.polylines(frame,np.int32([box1]), True ,(0,0,255),2  )
    else:
        cv2.polylines(frame,np.int32([box1]),True,(0,255,0), 2)

    if score[1] == 0: 
        cv2.polylines(frame,np.int32([box2]), True ,(0,0,255),2  )
    else:
        cv2.polylines(frame,np.int32([box2]),True,(0,255,0), 2)



    

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # Se a tecla 'q' for pressionada, finaliza com o loop
    if key == ord("q"):
        break

    if key == ord('p'):
        hist = cv2.calcHist([a[1]], [0], None, [256], [0,256])
        plt.plot(hist)
        plt.show()


    i += 1 









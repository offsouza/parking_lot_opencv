import cv2  
import imutils
from imutils import perspective 
from imutils.video import count_frames
import numpy as np
from matplotlib import pyplot as plt 
import time

WIDTH, HEIGHT = 100, 100 



def getSpotsCoordiantesFromImage(img, num_space) :
    #coordinate_lists has this format[ [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], [], [] ]
    coordinate_lists = []
    spots_index_list = []
    for i in range(num_space):
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])
        #we need 4 points to get rectangle
        print("Please click 4 points for parking lot in clock direction", i)
        coordinate = plt.ginput(4)
        print("clicked points coordinate are ", coordinate)
        coordinate_lists.append(coordinate)
        spots_index_list.append(i)
    plt.close()
    #saveSpotsCoordinates(coordinate_lists)
    #saveSpotsIndex(spots_index_list)
    return coordinate_lists


''' get rotate rectangle '''
def getRotateRect(img, cooridnate_lists):
    #warped image list is the list with warper images
    warped_img_lists = []
    i = 0 
    #every time we process one coordinates
    for coordinate in cooridnate_lists :
        warped = perspective.four_point_transform(img, coordinate)
        warped_resize = cv2.resize(warped, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        
        # plt.imshow(warped, cmap = 'gray', interpolation = 'bicubic')
        # plt.xticks([]), plt.yticks([])
        # plt.show()
        cv2.imshow("resize %d"%i, warped_resize)
        
        warped_img_lists.append(warped_resize)

        i+=1
    return warped_img_lists


def auto_canny(image, sigma=0.1):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, 200, 250)
 
    # return the edged image
    return edged


def SVM(frame):

    hist = cv2.calcHist([a[1]], [0], None, [256], [0,256])









path = 'videos/p2.mp4'
vs = cv2.VideoCapture(path)
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

num_frames = count_frames(path)
print(num_frames)


check = True
i = 0 

while i < num_frames:
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=450)  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray ", gray )

    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    cv2.imshow("gau ", gray )
    

    '''if check == True:
        boxes = getSpotsCoordiantesFromImage(frame, 2)
        print(boxes)
        check = False'''

    box1 = np.array([(155.5483870967742, 190.4596774193548), (206.35483870967744, 194.08870967741933), (196.375, 227.6572580645161), (139.21774193548384, 226.74999999999994)])
    box2 = np.array ([(131.05241935483872, 227.6572580645161), (82.06048387096773, 218.58467741935482), (112.90725806451611, 187.7379032258064), (151.9193548387097, 190.4596774193548)])
    boxes = [box1, box2]

    img_resize = getRotateRect(gray, boxes)

    canny = auto_canny(img_resize[1])
    cv2.imshow("Canny", canny)

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









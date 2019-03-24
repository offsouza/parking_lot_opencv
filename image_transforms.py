import cv2  
import imutils
from imutils import perspective 
#from imutils.video import count_frames
import numpy as np
from matplotlib import pyplot as plt 
import glob

                
#WIDTH, HEIGHT = 100, 100 


class utils(object):

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


    ''' rotaciona imagens '''
    def getRotateRect(img, cooridnate_lists, WIDTH = 100, HEIGHT = 100):
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


    def load_images_from_path(path):

        lista_imagens = []

        img_list = []
        for files in glob.glob(folderPath + "/*.jpg"):
            img = cv2.imread(files)
            img_list.append(img)
        return img_list
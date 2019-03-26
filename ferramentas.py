import cv2  
import imutils
from imutils import perspective 

import numpy as np
from matplotlib import pyplot as plt 
import glob


class image_utils():

    def __init__(self):
        pass

    def extract_features(self, img_resize):

        lista_globals = []

        for image in img_resize:

            hist = cv2.calcHist([image], [0], None, [256], [0,256])
            hist = hist.flatten()

            edged = cv2.Canny(image, 200, 250)
            edged = edged.flatten()

            flat = image.flatten()

            global_feature = np.hstack([hist, edged, flat])
            #global_feature = flat
            #print(global_feature.shape)
            lista_globals.append(global_feature)


        #print(lista_globals.shape)
        lista_globals = np.asarray(lista_globals)
        #print(lista_globals.shape)


        return lista_globals

            
            

    def getSpotsCoordiantesFromImage(img, num_space) :
        #coordinate_lists has this format[ [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], [], [] ]
        coordinate_lists = []
        spots_index_list = []
        for i in range(num_space):
            plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
            plt.xticks([]), plt.yticks([])
            #we need 4 points to get rectangle
            print(" Por favor clique nos 4 pontos da vaga em sentido horario", i)
            coordinate = plt.ginput(4)
            print("Os ponto selecionados são: ", coordinate)
            coordinate_lists.append(coordinate)
            spots_index_list.append(i)
        plt.close()
        
        return coordinate_lists


    ''' rotaciona imagens '''
    def getRotateRect(img, cooridnate_lists, WIDTH = 100, HEIGHT = 100):
        
        warped_img_lists = []
        i = 0 
        
        for coordinate in cooridnate_lists :
            warped = perspective.four_point_transform(img, coordinate)
            
            warped_resize = cv2.resize(warped, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            
            # plt.imshow(warped, cmap = 'gray', interpolation = 'bicubic')
            # plt.xticks([]), plt.yticks([])
            # plt.show()
            cv2.imshow("Vaga - %d"%i, warped_resize)
            
            warped_img_lists.append(warped_resize)

            i+=1
        return warped_img_lists


    def load_image_from_path(self, path):

        lista_imagens = []

        img_list = []
        for files in glob.glob(path + "/*.jpg"):
            img = cv2.imread(files)

            img = self.transform_image(img)            
            img_list.append(img)

        return img_list


    def load_imagens_keras_with_labels(path):
        from keras.preprocessing.image import ImageDataGenerator 

        train = ImageDataGenerator()
        base_train = train.flow_from_directory(path, class_mode='binary')

    def transform_image(self, image, WIDTH = 100, HEIGHT = 100):

        img = image

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7, 7), 0)    
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

        return img
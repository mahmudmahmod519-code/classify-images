#models
from sklearn.tree import DecisionTreeClassifier

#tools
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import os
import re

class Classify_Image:
  __output={"human":0,"animal":1,"plant":2}

  def __init__(self,folder_human_path,folder_animal_path,folder_planet_path):
    self.NEW_SIZE_IMAGE=(128,128)
    self.__model=DecisionTreeClassifier()
    self.__x_train=[]
    self.__y_train=[]
    folder_human=os.listdir(folder_human_path)
    folder_animal=os.listdir(folder_animal_path)
    folder_planet=os.listdir(folder_planet_path)
    
    for i in folder_human:
      image=cv2.imread(f'{folder_human_path}/{i}')
      self.__main_process(image)
      self.__y_train.append(self.__output['human'])

    for i in folder_animal:
      image=cv2.imread(f'{folder_animal_path}/{i}')
      self.__main_process(image)
      self.__y_train.append(self.__output['animal'])

    for i in folder_planet:
      image=cv2.imread(f'{folder_planet_path}/{i}')
      self.__main_process(image)
      self.__y_train.append(self.__output['plant'])

  
  def __extract_color(self,image):
    image=cv2.resize(image,self.NEW_SIZE_IMAGE)
    if image.shape[-1]==3 :
      feature=[]
      for i in range(3):
        hist=cv2.calcHist([image],[i],None,[256],[0,256])
        feature.extend(hist.flatten())
      return np.array(feature)
    else:
      hist=cv2.calcHist([image],[0],None,[256],[0,256])
      return np.array(hist.flatten())

  def __extract_edge(self,image):
    image=cv2.resize(image,self.NEW_SIZE_IMAGE)
    image=cv2.Canny(image,50,200)
    return np.array(image.flatten())


  def __main_process(self,image):
      image=cv2.resize(image,self.NEW_SIZE_IMAGE)
      gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
      fetau=np.concatenate([self.__extract_edge(gray_image),self.__extract_color(image)])
      self.__x_train.append(fetau)

  def fit_model(self):
    self.__model.fit(self.__x_train,self.__y_train)


  def real_time(self,path_image):
    #read image
    image=cv2.imread(path_image)
    #const size for each image
    image=cv2.resize(image,self.NEW_SIZE_IMAGE)
    # gray image to get edges
    gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #concatenate all feature and 
    fetau=np.concatenate([self.__extract_edge(gray_image),self.__extract_color(image)])
    return self.__model.predict([fetau])

  def test_accuorce(self,paths,y_test):
    y_pred=[]
    for path_image in paths:
      y_pred.append(self.test_model(path_image))
    return accuracy_score(y_test,y_pred)


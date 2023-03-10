import os
import cv2

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    # raise NotImplementedError("To be implemented")
    dataset = [] # list

    # face
    folder = dataPath + '/face/' # 連到face資料夾
    for image in os.listdir(folder): # listdir : return list of dir in folder
      dataset.append(cv2.imread(folder + image, cv2.IMREAD_GRAYSCALE), 1)

    # non-face
    folder = dataPath + '/non-face/'
    for image in os.listdir(folder):
      dataset.append(cv2.imread(folder + image, cv2.IMREAD_GRAYSCALE), 0)!
    
    # End your code (Part 1)
    return dataset
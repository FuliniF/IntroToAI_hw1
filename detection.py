import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")

    # save data into people list
    detectData = list(open(dataPath, "r"))
    line = 0
    while(line < len(detectData)): # 逐行執行
      filename, people_num = detectData[line].split()
      line += 1
      people = []
      for i in range(int(people_num)):
        x, y, width, height = detectData[line].split()
        people.append([int(x), int(y), int(width), int(height)])
        line += 1

      # reach image and resize it
      image = cv2.imread('data/detect/' + filename)
      grayimage = cv2.imread('data/detect/' + filename, cv2.IMREAD_GRAYSCALE)
      for i in range(int(people_num)):
        grayimageface = grayimage[people[i][1]:people[i][1]+people[i][3], people[i][0]:people[i][0]+people[i][2]]
        grayimageface = cv2.resize(grayimageface, (19,19))
        
        # detect and draw box
        if clf.classify(grayimageface) == 1:
          cv2.rectangle(image, (people[i][0], people[i][1]), (people[i][0]+people[i][2], people[i][1]+people[i][3]), (0, 255, 0), 3)
        else:
          cv2.rectangle(image, (people[i][0], people[i][1]), (people[i][0]+people[i][2], people[i][1]+people[i][3]), (0, 0, 255), 3)

      # make the picture clearer and show it
      kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
      image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
      plt.axis("off")
      plt.imshow(cv2.cvtColor(image_sharp, cv2.COLOR_BGR2RGB))
      plt.show()

    # End your code (Part 4)

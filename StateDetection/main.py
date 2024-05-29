import cv2
import hugh_circle as hugh
import numpy as np
import cProfile
import re


def preProcess(frame):
    pass

    new_frame = cv2.resize(frame.copy(), (0, 0), fx = 8, fy = 8)

    lower = np.array([0, 40, 40])
    upper = np.array([97, 255, 255])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(new_frame, new_frame, mask=mask)

    return result

    


def main():
    pass

    #get number of images in directory
    #load images
    min = 31
    max = 54
    path = 'C:/Users/tomkr/OneDrive/Documents/Uni/2024/COSC428/TrafficLightDetection/testImages/State/'

    for i in range(min, max):
        #print(i)
        full_path = path + 'state' + str(i) + '.jpg'
        img = cv2.imread(full_path)
        cv2.imshow("frame", cv2.resize(img, (0, 0), fx = 8, fy = 8))
        cv2.waitKey(1)
        pre_frame = hugh.preProcess(img)

        circle, state = hugh.hugh(img, pre_frame)
        print(state + " : " + str(i))
        pass
        #cv2.imshow("frame", cv2.resize(circle, (0, 0), fx = 8, fy = 8))
        #cv2.waitKey(1)

    # Load classifications

    cv2.waitKey(0)
    
main()
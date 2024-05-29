import cv2
import numpy as np

"""
Uses hugh circle algorithm to find the state of a traffic light
Pre processes frames with HSV value filtering for more acurate results
Assumes each frame input is a traffic light but hsv filtering usually removes
any non traffic light results

"""

SCALE = 8

index = 0
hugh_index = 0
img = 0
output_index = 0

DEBUG = False
DEBUG_ERRORS = False   # Outputs any errors
OUTPUT_CIRCLES = False # Displays the circle
SAVE_CIRCLES = False


def preProcess(frame):
    """ Scales up the traffic light
    Performs HSV filtering to remove any potential noise
    """

    new_frame = cv2.resize(frame.copy(), (0, 0), fx = 8, fy = 8)

    lower = np.array([0, 60, 60])
    upper = np.array([97, 255, 255])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(new_frame, new_frame, mask=mask)

    return result


def hugh(frame, preproc):
    global hugh_index, DEBUG

    img = preproc.copy()

    circles = hough_circle(img)

    if circles is not None:

        if OUTPUT_CIRCLES:
            copy_of_frame = cv2.resize(frame.copy(), (0, 0), fx = SCALE, fy = SCALE)
            for i in circles[0,:]:
                # Draw the outer circle
                cv2.circle(copy_of_frame,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
                # Draw the center of the circle
                cv2.circle(copy_of_frame,(int(i[0]),int(i[1])),2,(255,0,255),3) 

            cv2.imshow("Circle", img)
            cv2.waitKey(1)


        detected_state = get_state(circles, frame)

        return frame, detected_state
    
    elif DEBUG_ERRORS: # No circles detected
        cv2.imwrite("NoCircle"+str(hugh_index) +".jpg", frame)
        hugh_index += 1


    av = get_average(img)
    if av[0] != av[0]:
        return frame, "No Light Detected"
    detected_state = color_from_ranges(av)

    return frame, detected_state


def hough_circle(frame):
    """  Runs hough circle algorithm and adjusts the parameters until
    just one circle is found
    """
    global index
    
    #blur image
    blur = cv2.GaussianBlur(frame, (9,9), 0)
    # Convert the image to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Starting parameters
    param1 = 45
    param2 = 20
    minRadius = 0
    maxRadius = 100

    # While loop will move thresholds around until one circle is found
    found = False
    cycles = 0
    MAX_CYCLES = 5

    while not found and cycles < MAX_CYCLES:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

        cycles += 1

        # Check the amount of circles detected and adjusts the parameters
        if circles is None:
            if param1 - param2 >10:
                param1 -= 3
            else:
                param1 -= 3
        else: 
            circles = np.uint16(np.around(circles))
            if circles is None:
                param1 -= 5
            elif len(circles[0,:]) > 1:
                param1 += 5
            else:
                return circles

    return circles



def get_average(frame):
    """ Gets the non-black average color within an image
    """
    a2D = frame.reshape(-1,frame.shape[-1])
    non_black_pixels = a2D[~np.all(a2D == [0, 0, 0], axis=1)]
    if non_black_pixels.size > 0:
        av = np.mean(non_black_pixels, axis=0)
        return av
    return np.array([0, 0, 0])


def color_from_ranges(av):
    """  Gets the state given the average colour from state ranges
    Yellow is not implemented and is always assumed to be red
    """
    
    red_lower       = np.array((40, 55, 111))
    red_upper       = np.array((175, 225, 260))
    yellow_lower    = np.array((0, 0, 0))
    yellow_upper    = np.array((0, 0, 0))
    green_lower     = np.array((60, 55, 5))
    green_upper     = np.array((230, 230, 111))
    if np.all(av >= red_lower) and np.all(av <= red_upper):
        return "Red"
    elif np.all(av >= green_lower) and np.all(av <= green_upper):
        return "Green"
    elif np.all(av >= yellow_lower) and np.all(av <= yellow_upper):
        return "Yellow"


    return "None"


def get_state(circle_coords, frame):
    """ Takes the coords of the light and 
    the frame to get the state 
    """
    global DEBUG

    try:
        # uf 
        if circle_coords is not None:
            
            # Gets the largest circle if there are multiple
            largest = None
            for i in circle_coords[0,:]:
                if largest is None:
                    largest = i
                else:
                    if i[2] > largest[2]:
                        largest = i

            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.circle(mask, (int(largest[0]/SCALE),int(largest[1]/SCALE)),int(largest[2]/SCALE), 255, -1)
            light_state = cv2.bitwise_and(frame, frame, mask=mask)


            av = get_average(light_state)

            # Checks if the average is none as when it is it doesn't equal itself
            if av[0] != av[0]:
                return "No Light Detected"

            # Returns the state
            return color_from_ranges(av)
        return "None"
    except:
        return "None"



import cv2
from ultralytics import YOLO
import time
import StateDetection.hugh_circle as state
import threading
import copy


STATE_OUTPUT = False
# Multithreading for state detection (BETA)
MULTITHREAD = False
# Input video 
INPUT_VIDEO = 'drive.mp4'
# Yolo model
YOLO_MODEL = 'models/57.pt'

def predict(chosen_model, img, classes=[], conf=0.5):
    """ Detects the traffic light using yolo and the model
    """
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf,  verbose = False)
    else:
        results = chosen_model.predict(img, conf=conf,  verbose = False)

    return results

# State buffer is an experiment for getting the state
# if no traffic light state is found, not sure how usefull it 
# would be for an actual self driving car but it makes the results
# look better 
class state_buffer:
    """ Acts as a circular buffer for states for storing states
    and getting the average state and confidence
    """
    def __init__(self, size = 10):
        self.size = size
        self.index = 0
        self.states = ["None" for i in range(self.size)]

    def get_average(self):
        """ Gets the average state with confidence
        state priority: red > none > yellow > green
        """
        none_state = 0
        red = 0
        yellow = 0
        green = 0

        for state in self.states:
            match state:
                case "Red":
                    red += 1
                case "Yellow":
                    yellow += 1
                    
                case "Green":
                    green += 1
                    
                case _:
                    none_state += 1

        state_list = [red,none_state, yellow, green]
        biggest = 0
        for index in range(len(state_list)):
            if state_list[index] > state_list[biggest]:
                biggest = index
        avg = state_list[biggest] / self.size
        match biggest:
            case 0:
                return "Red: " + f"{avg:.2f}"
            case 1:
                return "None: " + f"{avg:.2f}"
            case 2:
                return "Yellow: " + f"{avg:.2f}"
            case 3:
                return "Green: " + f"{avg:.2f}"

    def add_state(self, state):
        """ Updates the oldest entry with new state and returns
        the state or the average if the state is None
        """

        self.states[self.index] = state
        self.index += 1
        if self.index >= self.size:
            self.index = 0

buffer = state_buffer()


path = 'testImages/out/'
state_path = 'testImages/State/'


num = 0


def detect_state(img, box1, box2, box3, box4):
    """Gets the preprocssed frame using the outline coords of the traffic light
    then runs state detection on the frame
    """
    global num, path, state_path

    # crops the frame to just the bounding box
    cropped = img[box2: box4, box1: box3]

    # Pre-processes the frame for state detection
    pre_proc = state.preProcess(cropped)

    # Gets the state of the cropped traffic light
    state_img, state_detected = state.hugh(cropped, pre_proc)

    # Saves the output (For debugging)
    if STATE_OUTPUT:
        num += 1
        p1 = path + "out" + str(num) + ".jpg"
        p2 = state_path + "state" + str(num) + ".jpg"
        #cv2.imwrite(p1, state_img)
        cv2.imwrite(p2, cropped)

    return state_detected

def detect_state_therading(img, box1, box2, box3, box4, state_thread, index):
    """
    """
    global num, path, state_path

    # crops the frame to just the bounding box
    cropped = img[box2: box4, box1: box3]



    pre_proc = state.preProcess(cropped)
    state_detected = state.hugh(cropped, pre_proc)

    if STATE_OUTPUT:
        num += 1
        p1 = path + "out" + str(num) + ".jpg"
        p2 = state_path + "state" + str(num) + ".jpg"
        #cv2.imwrite(p1, cropped)
        cv2.imwrite(p2, cropped)

    state_thread[index] = state_detected
    pass


def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    """ Predicts the bounding box of the traffic light
    """
    global num
    # Get list of coords of predicted traffic lights
    results = predict(chosen_model, img, classes, conf=conf)

    num_thread = 0
    thread_queue = []
    thead_objects = ['None'] * 20
    index = 0



    for result in results:
        for box in result.boxes:
            box1 = int(box.xyxy[0][0])
            box2 = int(box.xyxy[0][1])

            box3 = int(box.xyxy[0][2])
            box4 = int(box.xyxy[0][3])


            detect_state_result = "none"
            if MULTITHREAD: # Creates a thread for each detected traffic light
                new = copy.deepcopy(detect_state_result)
                thead_objects.append(new)
                thread = threading.Thread(target=detect_state_therading, args=(img, box1, box2, box3, box4, thead_objects, index))
                thread_queue.append(thread)
                thread.start()
            else:
                # Draw rectangle around traffic light 
                cv2.rectangle(img, (box1, box2),
                              (box3, box4), (255, 0, 0), 1)
                # Get state of traffic light
                state_detected = detect_state(img, box1, box2, box3, box4)
                # Add state to box
                cv2.putText(img, f"{state_detected}", (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            index += 1

    if MULTITHREAD: 
    # Waits for each state detection thread and writes the output to the screen
        for item in thread_queue:
            item.join()

        index = 0
        for result in results:
            for box in result.boxes:
                cv2.rectangle(img, (box1, box2),
                            (box3, box4), (255, 0, 0), 1)

                cv2.putText(img, f"{thead_objects[index][1]}", (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                index += 1
    
    return img, results


def main():
    # What fps to process at
    # 0 for unlimited
    PROCESS_FPS = 0
    if PROCESS_FPS == 0:
        frame_delay = 0
    else:
        frame_delay = 1 / PROCESS_FPS

    # Input source/video
    video = cv2.VideoCapture(INPUT_VIDEO)
    # Yolo model
    model = YOLO(YOLO_MODEL)

    img_counter = 0

    start_time = time.time()

    while(video.isOpened()):
        
        img_counter += 1
        ret, frame = video.read()

        if ret:
            # Resize the frame
            processed_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            result_img = predict_and_detect(model, processed_frame, classes=[], conf=0.5)

            # Display the output
            cv2.imshow("Output", result_img[0])
            cv2.waitKey(1)

        else:
            break

    # Close video and windows
    video.release()
    cv2.destroyAllWindows()

    # Outputs the total time and fps average
    end_time = time.time()
    print("Total Time : {:.3f}".format(end_time - start_time))
    print("Total Frames: {}".format(img_counter))
    fps = img_counter / (end_time - start_time)
    print("FPS: {:.2f}".format(fps))
            




main()
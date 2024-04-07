from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
import tkinter as tk
from tkinter import messagebox
import threading
import webbrowser
import pygame

last_warning_time = 0  # Keeps track of when the last warning was played
warning_cooldown = 2  # Cooldown period in seconds between warnings

def play_warning_sound(sound):
    global last_warning_time
    current_time = time.time()
    if current_time - last_warning_time >= warning_cooldown:
        sound.play()
        last_warning_time = current_time

def wait_for_blink_response():
    global prompt_shown
    prompt_shown = True
    
    left_blink_counter = 0
    right_blink_counter = 0
    response = None

    while response is None:
        frame = vs.read()
        frame = imutils.resize(frame, width=1024, height=576)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            if leftEAR < EYE_AR_THRESH:
                left_blink_counter += 1
            if rightEAR < EYE_AR_THRESH:
                right_blink_counter += 1

            if left_blink_counter >= EYE_AR_CONSEC_FRAMES:
                response = False
                break
            if right_blink_counter >= EYE_AR_CONSEC_FRAMES:
                response = True
                break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    prompt_shown = False
    return response

# Add the TensorFlow model for object detection
def load_object_detection_model():
    config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weights_path = 'frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    return net

# Function to read the COCO names
def read_coco_names(file_path):
    with open(file_path, 'rt') as f:
        return f.read().rstrip('\n').split('\n')

# Function to detect objects using OpenCV DNN
def detect_objects(frame, net, classNames):
    class_ids, confidences, bbox = net.detect(frame, confThreshold=0.5)
    warning_objects = []
    if len(class_ids) > 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), bbox):
            class_name = classNames[class_id - 1]
            if class_name in ["cell phone", "bottle"]:  # Added "smoking" to the list
                warning_objects.append(class_name)
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"{classNames[class_id-1]}: {int(confidence*100)}%", (box[0]+10, box[1]+30), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
    return warning_objects


# Add a global variable to keep track of the prompt window
prompt_shown = False
cell_phone_detected = False
drinking_detected = False

def show_rest_prompt():
    global prompt_shown

    # Check if the prompt is already being shown
    if prompt_shown:
        return

    def on_prompt_response(response):
        global prompt_shown
        if response:
            webbrowser.open("https://www.google.com/maps/search/hotels+near+me")
        # Ensure the root window is destroyed and the flag is reset
        root.destroy()
        prompt_shown = False

    # Set the flag indicating the prompt is being shown
    prompt_shown = True
    root = tk.Tk()
    root.withdraw()
    root.after(0, lambda: on_prompt_response(
        messagebox.askyesno("Take a Rest", "You seem tired. Do you want to find a hotel near you?")
    ))
    # This ensures the root window is destroyed if the window is closed using the window manager
    root.protocol("WM_DELETE_WINDOW", lambda: on_prompt_response(False))
    root.mainloop()


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("[INFO] initializing camera...")

pygame.mixer.init()  # Initialize the mixer
focus_sound = pygame.mixer.Sound('focus.mp3')  # Load your sound file
drinkings= pygame.mixer.Sound('drinking.mp3')
cells= pygame.mixer.Sound('cellphone.mp3')

def play_focus_sound():
    if not pygame.mixer.get_busy():  # Check if any sound is playing
        focus_sound.play()  # Play the sound if nothing is playing

def stop_focus_sound():
    focus_sound.stop() 


vs = VideoStream(src=0).start()

time.sleep(2.0)

frame_width = 1024
frame_height = 576

image_points = np.array([
    (359, 391),     
    (399, 561),    
    (337, 297),   
    (513, 301),     
    (345, 465),     
    (453, 469)      
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.74
EYE_AR_CONSEC_FRAMES = 3

LONG_BLINK_THRESHOLD = 20  

COUNTER = 0
LONG_BLINK_COUNTER = 0

yawn_counter = 0
YAWN_THRESH = 15

(mStart, mEnd) = (49, 68)

net = load_object_detection_model()
classNames = read_coco_names('coco.names')

while True:
  
    frame = vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    rects = detector(gray, 0)

    warning_objects = detect_objects(frame, net, classNames)
    
    # Check for specific objects and display warnings
    if "cell phone" in warning_objects and not cell_phone_detected:
        cv2.putText(frame, "Cell Phone Detected!", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        play_warning_sound(cells)  # Use the wrapper function to play the sound
        cell_phone_detected = True
    elif "cell phone" not in warning_objects:
        cell_phone_detected = False  # Reset the flag if the cell phone is no longer detected

    if "bottle" in warning_objects and not drinking_detected:
        cv2.putText(frame, "Possible Drinking Detected!", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        play_warning_sound(drinkings) 
        drinking_detected = True
    elif "bottle" not in warning_objects:
        drinking_detected = False
    
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for rect in rects:
 
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
   
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            LONG_BLINK_COUNTER += 1  # Increment the long blink counter as well
    
            if LONG_BLINK_COUNTER >= LONG_BLINK_THRESHOLD:
                play_focus_sound()  # Consider playing the sound on long blink threshold
                cv2.putText(frame, "Long Blink Detected!", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                threading.Thread(target=show_rest_prompt, daemon=True).start()
            
            if COUNTER >= EYE_AR_CONSEC_FRAMES and LONG_BLINK_COUNTER+5 <= LONG_BLINK_THRESHOLD :
                cv2.putText(frame, "Eyes Closed!", (500, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                stop_focus_sound() 
            COUNTER = 0
            LONG_BLINK_COUNTER = 0 

        if yawn_counter >= YAWN_THRESH:
            yawn_counter = 0  
            play_focus_sound()
            threading.Thread(target=show_rest_prompt, daemon=True).start()

        mouth = shape[mStart:mEnd]

        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (750, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (900, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawn_counter += 1

        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                image_points[0] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point, 
            end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

        if head_tilt_degree:
            cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF



    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

import cv2
import dlib
import numpy as np
import math
import time
import threading
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import argparse
from ultralytics import YOLO

model1 = YOLO('best_mobile.pt')
model2 = YOLO('best_yawn.pt')

threshold_m = 0.75
threshold_y = 0.5

ALERTNESS_METER = 0
ALERTNESS_THRESHOLD = 50
EYES_CLOSED_COUNTER= 0

class EARFilter:
    def __init__(self, window_size=5):
        self.values = []
        self.window_size = window_size
    
    def update(self, new_value):
        self.values.append(new_value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)

path="alert_sound.wav"

def sound_alarm(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    
def play_alarm_sound(path):
    sound_alarm(path)
    time.sleep(5)  
    pygame.mixer.music.stop()
    
def eye_aspect_ratio(eye):
	
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def mouth_aspect_ratio(mouth):
    
    X = dist.euclidean(mouth[2], mouth[10])  
    Y = dist.euclidean(mouth[4], mouth[8])   

    Z = dist.euclidean(mouth[0], mouth[6])   

    mar = (X + Y) / (2.0 * Z)
    return mar

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")

args = vars(ap.parse_args())

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48

TALKING_THRESH = 0.2  
MIN_TALKING_EVENTS = 2  
mouthOpen = False       
mouthEvents = 0        

COUNTER = 0
ALARM_ON = False

NOT_LOOKING_TIMER = 0
EYE_CLOSED_TIME = 1.0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
start_time = time.time()



focal_length = frame.shape[1]
center = (frame.shape[1]/2, frame.shape[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

direction = "Front"

ear_filter = EARFilter()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    results1 = model1(frame)[0]
    results2 = model2(frame)[0]
    
    def draw_detections(frame, x1, y1, x2, y2, score, class_name):
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, class_name, (int(x1), int(y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f'Confidence: {score:.2f}', (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        print(f"Detected: {class_name}, Score: {score}")
    
    for result in results1.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_name = results1.names[int(class_id)].upper()
        if score > threshold_m:
            draw_detections(frame, x1, y1, x2, y2, score, class_name)
        
    for result in results2.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_name = results2.names[int(class_id)].upper()
        if score > threshold_y:
            draw_detections(frame, x1, y1, x2, y2, score, class_name)
            
    end_time = time.time()  
    FPS = 1.0 / (end_time - start_time) if end_time > start_time else 0.0

    
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        shape = face_utils.shape_to_np(landmarks)


        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (51, 153, 255), -1)

        image_points = np.array([
                                (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
                                (landmarks.part(8).x, landmarks.part(8).y),     # Chin
                                (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
                                (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
                                (landmarks.part(48).x, landmarks.part(48).y),     # Left Mouth corner
                                (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
                            ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        mouth = shape[48:68]  
        mar = mouth_aspect_ratio(mouth)

        dist_coeffs = np.zeros((4,1)) 
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rmat, _ = cv2.Rodrigues(rotation_vector)
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(np.concatenate((rmat, translation_vector), axis=1))

        pitch, yaw, roll = [math.radians(_[0]) for _ in euler_angle]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        if yaw > 30:
            new_direction = "Right"
        elif yaw < -30:
            new_direction = "Left"
        elif pitch > 15:
            new_direction = "Up"
        elif pitch < -15:
            new_direction = "Down"
        else:
            new_direction = "Front"

        if new_direction != direction:
            direction = new_direction
            
        leftEye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        rightEye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
    
        ear = (leftEAR + rightEAR) / 2.0
        ear = ear_filter.update(ear)  
        
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255,0,0), 2)

        cv2.putText(frame, "Looking: " + direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (102, 102, 255), 1)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
        height = frame.shape[0]
        
        if mar > TALKING_THRESH and not mouthOpen:
            mouthOpen = True
            mouthEvents += 1
        elif mar <= TALKING_THRESH and mouthOpen:
            mouthOpen = False
            mouthEvents += 1

    # Check if the number of events indicates talking
        if mouthEvents >= MIN_TALKING_EVENTS:
            isTalking = True
            mouthEvents = 0  # Reset the counter after detecting talking
        else:
            isTalking = False
            
        if isTalking:
            cv2.putText(frame, "Talking", (500, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Not Talking", (500, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
        cv2.putText(frame, "No Seatbelt !!", (300, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

     
        if new_direction != "Front":
            NOT_LOOKING_TIMER += 1
            if NOT_LOOKING_TIMER >= 5 * FPS:  
                ALERTNESS_METER += 1
                NOT_LOOKING_TIMER = 0
        else:
            NOT_LOOKING_TIMER = 0

        if "OPEN-MOUTH" in [class_name for result in results2.boxes.data.tolist()]:
            ALERTNESS_METER += 1

        if ear < EYE_AR_THRESH:
            EYES_CLOSED_COUNTER += 1
            if EYES_CLOSED_COUNTER >= EYE_CLOSED_TIME * FPS:  
                ALERTNESS_METER += 1
                EYES_CLOSED_COUNTER = 0
        else:
            EYES_CLOSED_COUNTER = 0

        if "CELL PHONE" in [class_name for result in results1.boxes.data.tolist()]:
            ALERTNESS_METER += 1
    
        else:
            ALARM_ON = False
            ALERTNESS_METER_POSITION = (10, height-30)  
        
            cv2.putText(frame, f"Alertness: {ALERTNESS_METER}", ALERTNESS_METER_POSITION, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
            ALERTNESS_BAR_POSITION = (10, height-20)  
            ALERTNESS_BAR_MAX_WIDTH = 200  

            alertness_bar_width = int(ALERTNESS_METER / ALERTNESS_THRESHOLD * ALERTNESS_BAR_MAX_WIDTH)
            cv2.rectangle(frame, ALERTNESS_BAR_POSITION, (ALERTNESS_BAR_POSITION[0] + ALERTNESS_BAR_MAX_WIDTH, ALERTNESS_BAR_POSITION[1] + 20), (255, 255, 255), -1)
            cv2.rectangle(frame, ALERTNESS_BAR_POSITION, (ALERTNESS_BAR_POSITION[0] + alertness_bar_width, ALERTNESS_BAR_POSITION[1] + 20), (0, 0, 0), -1)

            cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(ear), (350, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
        if ALERTNESS_METER >= ALERTNESS_THRESHOLD:
            ALERTNESS_METER = 0
            alarm_thread = threading.Thread(target=play_alarm_sound, args=(path,))
            alarm_thread.start()
    
    else:
        print("No faces detected.")
                
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
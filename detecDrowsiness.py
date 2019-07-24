#Name: Tanuja Joshi
#Section: CS-D
#Roll No: 22
#DROWSY DRIVER DETECTION SYSTEM

import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import playsound

#For eye aspect ratio to indicate blink
EYE_AR_THRESH = 0.18
#Number of consecutive frames the eye must be below the threshold for to set off the alarm
EYE_AR_CONSEC_FRAMES = 7
#For mouth aspect ratio to indicate jaw opening 
MOUTH_AR_THRESH = 0.30

SHOW_POINTS_FACE = False
SHOW_CONVEX_HULL_FACE = False
SHOW_INFO = False
#To store calculated value of eye_aspect_ratio
ear = 0
#To store calculated value of mouth_aspect-ratio
mar = 0
#To Count Number Of Frames for eyes 
COUNTER_FRAMES_EYE = 0
#To Count Number Of Frames for mouth
COUNTER_FRAMES_MOUTH = 0
#To Count eyes blink
COUNTER_BLINK = 0
#To Count Mouth Opening
COUNTER_MOUTH = 0

#Funtion to Calculate eye_aspect_ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

#Funtion To Calculate mouth_aspect_ratio
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[5], mouth[8])
    B = dist.euclidean(mouth[1], mouth[11])	
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C) 



#For Video Capturing from video files or camera
videoSteam = cv2.VideoCapture(0)
#To Capture frame by frame
ret, frame = videoSteam.read()
size = frame.shape

#Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()
#Create the facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#Grab Indexes Of facial landmark for left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

model_points = np.array([(0.0, 0.0, 0.0),
                         (0.0, -330.0, -65.0),        
                         (-225.0, 170.0, -135.0),     
                         (225.0, 170.0, -135.0),      
                         (-150.0, -150.0, -125.0),    
                         (150.0, -150.0, -125.0)])

focal_length = size[1]
center = (size[1]/2, size[0]/2)

camera_matrix = np.array([[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double")

dist_coeffs = np.zeros((4,1))

t_end = time.time()
while(True):
    #grab the frame from the camera and convert it to grayscale channels
    ret, frame = videoSteam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces in grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy
		  # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, and jaw coordinates 
        #then use the coordinates to compute the eye aspect ratio for both eyes and mouth aspect ratio for mouth
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        jaw = shape[48:61]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye) 
        #Average
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(jaw)

        image_points = np.array([
                                (shape[30][0], shape[30][1]),
                                (shape[8][0], shape[8][1]),
                                (shape[36][0], shape[36][1]),
                                (shape[45][0], shape[45][1]),
                                (shape[48][0], shape[48][1]),
                                (shape[54][0], shape[54][1])
                                ], dtype="double")


        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        if SHOW_POINTS_FACE:
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        
        if SHOW_CONVEX_HULL_FACE: 
            # compute the convex hull for the left and right eye and mouth, then
		      # visualize each of the eyes and mouth
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            jawHull = cv2.convexHull(jaw)
            cv2.drawContours(frame, [leftEyeHull], 0, (255, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], 0, (255, 255, 255), 1)
            cv2.drawContours(frame, [jawHull], 0, (255, 255, 255), 1)


        if COUNTER_BLINK > 10 or COUNTER_MOUTH > 2:
            cv2.putText(frame, "Alert!", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            playsound.playsound('beep1.mp3')
            
            
		# check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER_FRAMES_EYE += 1
            #If eyes is closed for sufficent number of frames then sound the alarm
            if COUNTER_FRAMES_EYE >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Sleeping Driver!", (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                playsound.playsound('beep1.mp3')
        else:
            if COUNTER_FRAMES_EYE > 2:
                COUNTER_BLINK += 1
            COUNTER_FRAMES_EYE = 0
        # check to see if the mouth aspect ratio is below the opening threshold, and if so, increment the mouth frame counter
        if mar >= MOUTH_AR_THRESH:
            COUNTER_FRAMES_MOUTH += 1
        else:
            if COUNTER_FRAMES_MOUTH > 5:
                COUNTER_MOUTH += 1
      
            COUNTER_FRAMES_MOUTH = 0
        
        if (time.time() - t_end) > 60:
            t_end = time.time()
            COUNTER_BLINK = 0
            COUNTER_MOUTH = 0
        
    if SHOW_INFO:
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (200, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Blinks: {}".format(COUNTER_BLINK), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Mouths: {}".format(COUNTER_MOUTH), (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    #Show the frame
    cv2.imshow("Output", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('p'):
        SHOW_POINTS_FACE = not SHOW_POINTS_FACE
    if key == ord('c'):
        SHOW_CONVEX_HULL_FACE = not SHOW_CONVEX_HULL_FACE
    if key == ord('i'):
        SHOW_INFO = not SHOW_INFO
    time.sleep(0.03)
    
videoSteam.release()  
cv2.destroyAllWindows()

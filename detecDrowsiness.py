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
eye_thresh = 0.18
#Number of consecutive frames the eye must be below the threshold for to set off the alarm
max_consec_frames = 7
#For mouth aspect ratio to indicate jaw opening 
mouth_thresh = 0.30

show_points = False
show_conv_hull = False
show_info = False
#To store calculated value of eye_aspect_ratio
ear = 0
#To store calculated value of mouth_aspect-ratio
mar = 0
#To Count Number Of Frames for eyes 
count_fr_eye = 0
#To Count Number Of Frames for mouth
count_fr_mouth = 0
#To Count eyes blink
blink_count = 0
#To Count Mouth Opening
yawn_count = 0

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

        if show_points:
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        
        if show_conv_hull: 
            # compute the convex hull for the left and right eye and mouth, then
		      # visualize each of the eyes and mouth
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            jawHull = cv2.convexHull(jaw)
            cv2.drawContours(frame, [leftEyeHull], 0, (255, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], 0, (255, 255, 255), 1)
            cv2.drawContours(frame, [jawHull], 0, (255, 255, 255), 1)


        if blink_count > 10 or yawn_count > 2:
            cv2.putText(frame, "Alert!", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            playsound.playsound('beep1.mp3')
            
            
		# check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
        if ear < eye_thresh:
            count_fr_eye += 1
            #If eyes is closed for sufficent number of frames then sound the alarm
            if count_fr_eye >= max_consec_frames:
                cv2.putText(frame, "Sleeping Driver!", (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                playsound.playsound('beep1.mp3')
        else:
            if count_fr_eye > 2:
                blink_count += 1
            count_fr_eye = 0
        # check to see if the mouth aspect ratio is below the opening threshold, and if so, increment the mouth frame counter
        if mar >= mouth_thresh:
            count_fr_mouth += 1
        else:
            if count_fr_mouth > 5:
                yawn_count += 1
      
            count_fr_mouth = 0
        
        if (time.time() - t_end) > 60:
            t_end = time.time()
            blink_count = 0
            yawn_count = 0
        
    if show_info:
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (200, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Blinks: {}".format(blink_count), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Mouths: {}".format(yawn_count), (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    #Show the frame
    cv2.imshow("Output", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('p'):
        show_points = not show_points
    if key == ord('c'):
        show_conv_hull = not show_conv_hull
    if key == ord('i'):
        show_info = not show_info
    time.sleep(0.03)
    
videoSteam.release()  
cv2.destroyAllWindows()

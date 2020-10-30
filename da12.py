from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame 
import time
import dlib
import cv2
import matplotlib.pyplot as plt
import requests
import json
import keyboard

#URL = 'https://www.sms4india.com/api/v1/sendCampaign'

pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

thresh = 0.25
frame_check = 10
MOUTH_AR_THRESH = 0.4

COUNTER = 0
mar = 0
ear = 0
flag = 0
X1 = []
X2 = []
i=0
j=0
COUNTER_FRAMES_MOUTH = 0
COUNTER_MOUTH = 0
COUNTER_BLINK = 0

SHOW_POINTS_FACE = False
face_cascade = cv2.CascadeClassifier("C://Users//harshad//Desktop//DDAS//Driver-Drowsiness-Alert-System//haarcascades//haarcascade_frontalface_default.xml")

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A+B) / (2*C)
	return ear

def mouth_aspect_ratio(mouth):
	A = distance.euclidean(mouth[5], mouth[8])
	B = distance.euclidean(mouth[1], mouth[11])	
	C = distance.euclidean(mouth[0], mouth[6])
	return (A + B) / (2.0 * C) 
	
'''def sendPostRequest(reqUrl, apiKey, secretKey, useType, phoneNo, senderId, textMessage):
  req_params = {
  'apikey':apiKey,
  'secret':secretKey,
  'usetype':useType,
  'phone': phoneNo,
  'message':textMessage,
  'senderid':senderId
  }
  return requests.post(reqUrl, req_params)'''


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
size = frame.shape

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





#time.sleep(2)
t_end = time.time()


def sound1():
	pygame.mixer.music.load('audio/alaram_1.mp3')
	
	
	pygame.mixer.music.play(1)
	time.sleep(1)
	
def sound2():
	pygame.mixer.music.load('audio/alaram2.mp3')
	
	
	pygame.mixer.music.play(1)
	time.sleep(5)

def sound4():
	pygame.mixer.music.load('audio/alaram4.mp3')
	
	
	pygame.mixer.music.play(1)
	time.sleep(5)
	
def sound5():
	pygame.mixer.music.load('audio/head.mp3')
	
	
	pygame.mixer.music.play(1)
	time.sleep(3)
	
def sound():
	pygame.mixer.music.load('audio/welcome.mp3')
	
	
	pygame.mixer.music.play(1)
	time.sleep(3)


	
	
	
i==0
j==0
while(True):
	
	#pygame.mixer.music.play(-1)
			
	
	ret, frame = video_capture.read()
	
	
		
		
	frame = cv2.flip(frame,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray, 0)

	face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in face_rectangle:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.putText(frame, "Face", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		
	for face in faces:
		shape = predictor(gray, face)
		shape = face_utils.shape_to_np(shape)
		
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		
		jaw = shape[48:61]
		mar = mouth_aspect_ratio(jaw)
		X1.append(ear)
		X2.append(mar)
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		jawHull = cv2.convexHull(jaw)
		
		
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

		p1 = (int(image_points[0][0]), int(image_points[0][1]))
		p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
		
		
		
		
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [jawHull], 0, (0, 255, 0), 1)
		cv2.line(frame, p1, p2, (255,255,255), 2)
		
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
			
		cv2.putText(frame, "Blinks: {}".format(COUNTER_BLINK), (500, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
			
		cv2.putText(frame, "Yawn: {}".format(COUNTER_MOUTH), (500, 50),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
			
		
			
	#msg=0
	
		#if COUNTER_BLINK > 7: 
			#cv2.putText(frame, "*************************ALERT!***********************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			#cv2.putText(frame, "*************************ALERT!***********************", (10,475),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
			#sound5()
		#else:
			#pygame.mixer.music.stop()
			
		if COUNTER_BLINK > 30: 
			cv2.putText(frame, "*************************ALERT!***********************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "*************************ALERT!***********************", (10,475),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			COUNTER_BLINK=0
			sound4()
		else:
			pygame.mixer.music.stop() 
			#pygame.mixer.music.load('audio/short_alarm.mp3')
			#pygame.mixer.music.play(-1)
			#msg=msg+1
		#if msg>0:
			#response = sendPostRequest(URL, 'K5IMGBX95S7VDLHK0X1D223FD7A2O2HQ', ' LONO95BVVT902AR3', 'stage', '8850731809', 'magamharshad@gmail.com', 'test' )
			
		if COUNTER_MOUTH > 2: 
			cv2.putText(frame, "*************************ALERT!***********************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "*************************ALERT!***********************", (10,475),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			#pygame.mixer.music.load('audio/power_alarm.wav')
			#pygame.mixer.music.play(-1)
			COUNTER_MOUTH=0
			sound2()
		else:
			pygame.mixer.music.stop() 
		
		#print(COUNTER_MOUTH)
		
		#if COUNTER_BLINK > 50 or COUNTER_MOUTH > 10: 
			#cv2.putText(frame, "Send Alert!", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		
		
		#if p2[1] > p1[1]*1.5: 
			#cv2.putText(frame, "Tend", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
		#if p2[1] > p1[1]*1: 
			#cv2.putText(frame, "Tend", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
		#if p2[1] > p1[1]*2.0: 
			#cv2.putText(frame, "Tend", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
		if p2[1] > p1[1]*1.0 and ear < thresh:
					if flag >= frame_check:
						cv2.putText(frame, "*************************Tend***********************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						sound5()
		
		if p2[1] > p1[1]*1.5:
			cv2.putText(frame, "*************************Tend***********************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			sound5()
		if ear < thresh:
			flag += 1
			#print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "*************************ALERT!***********************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "*************************ALERT!***********************", (10,475),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#pygame.mixer.music.load('audio/alert.wav')
				#pygame.mixer.music.play(-1)
				sound1()
				#msg=msg+1
		#if msg>0:
			#response = sendPostRequest(URL, 'K5IMGBX95S7VDLHK0X1D223FD7A2O2HQ', ' LONO95BVVT902AR3', 'stage', '8850731809', 'magamharshad@gmail.com', 'alert')
				
		else:
			pygame.mixer.music.stop()
			if flag >= 1:
				COUNTER_BLINK += 1
			flag = 0
			
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
	
    
	cv2.imshow('Video', frame)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break
		
	if i==0:
		sound()
	i=i+1
	

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(X1)
plt.title('ear graph')
#plt.show()
#ax.legend()
fig.savefig('ear3.png')  

fig1 = plt.figure()
ax1 = plt.subplot(111)
ax1.plot(X1)
plt.title('mar graph')
#plt.show()
#ax1.legend()
fig.savefig('mar3.png')   
		
		
video_capture.release()
cv2.destroyAllWindows()		
		
		







# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
fromaddr = "testcase30103@gmail.com"
toaddr = "akshay.jangid03@gmail.com"

t=0
tr=50


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] starting video stream...")
time.sleep(2.0)
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))
for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	tr=tr+1
	
	frame = frame1.array
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			idx = int(detections[0, 0, i, 1])
			if(idx!=15):
				continue
			else:	
				print('found human')
				if(tr-t>10):
					print('sending mail')
					camera.capture("/home/pi/a1.jpeg")
					msg = MIMEMultipart() 
					msg['From'] = fromaddr 
					msg['To'] = toaddr 
					msg['Subject'] = "test1"
					body = "body"
					msg.attach(MIMEText(body, 'plain')) 
					filename = "a1.jpeg"
					attachment = open("/home/pi/a1.jpeg", "rb") 
					p = MIMEBase('application', 'octet-stream') 
					p.set_payload((attachment).read()) 
					encoders.encode_base64(p) 
					p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
					msg.attach(p) 
					s = smtplib.SMTP('smtp.gmail.com', 587)  
					s.starttls() 
					s.login(fromaddr, "akshaydikshit") 
					text = msg.as_string() 
					s.sendmail(fromaddr, toaddr, text)  
					s.quit() 
					print('mail sent')
					t=tr
				rawCapture.truncate(0)
	

	#cv2.imshow("Frame", frame)    
	rawCapture.truncate(0) 
	key = cv2.waitKey(1) & 0xFF
        
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	#fps.update()

# stop the timer and display FPS information
#fps.stop()
#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
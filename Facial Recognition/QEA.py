import glob
import numpy as np
import cv2 as cv
import sys

#Matrix Inversion Function:
def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]

#Setting the Print Limit to Maximum. Makes the analysis of the code easier.
np.set_printoptions(threshold=sys.maxsize)

#Defining the path of the Photos. The Not Smily and Smily photos are stored in different folders so there is no mistake in the readings
path1 = glob.glob("Photosp/*.jpg")
path2 = glob.glob("Photosn/*.jpg")
cv_img = [] #Individual Images Stored in the Array
Matrix = np.zeros((625, (len(path1) + len(path2)))) #625 to 651 matrix to train the model
a = 0
#For loops to get all the image data in the array
for img in path1:
    n = cv.imread(img)
    n = cv.cvtColor(n, cv.COLOR_BGR2GRAY)
    n = cv.resize(n, (25,25))
    cv_img.append(n)
    Matrix[:, a] = cv.resize(n, (625, 1))
    a += 1

for img in path2:
    n = cv.imread(img)
    n = cv.cvtColor(n, cv.COLOR_BGR2GRAY)
    n = cv.resize(n, (25,25))
    cv_img.append(n)
    Matrix[:, a] = cv.resize(n, (625, 1))
    a += 1

#Assigning the smiling values for each image by creating a 'test' object
test = np.zeros((651,1))
test[0:209] = np.ones((209,1))
print(test) #Print to analyze for any mistakes

Matrix = np.swapaxes(( 1 - (Matrix/255)), 0, 1) #Getting a transpose of the matrix because the initial values are reverse.


wd =  np.dot(inv(np.dot(np.swapaxes(Matrix, 0, 1), Matrix)), np.dot(np.transpose(Matrix), test)) #The result 625 to 1 matrix that will be used to find the smiling value.
print(wd) #Checking if it is the correct value.

show = cv.resize(wd, (25,25))
cv.imshow('Eigen', show)

#Live Facial Recognition Part starts from here:
capture = cv.VideoCapture(0) #Assigning the video capture to the webcam.
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml') #This is the only code I didnt write from scratch:
#This implements a haar cascade to determine the position of the face.

while True: #While Loop for the live recognition.
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 5)
        frame_p = frame[(x+20):(x+w-20),(y+20):(y+h-20)] #The following code resizes the frame to include only the face and then sizes it to a 625 to 1 matrix to find the smiling factor.
        frame_p = cv.cvtColor(frame_p, cv.COLOR_BGR2GRAY)
        cv.imshow('Web Cam 2', frame_p) #Opens a window to represent the steps of the resizing
        frame_p = cv.resize(frame_p, (25,25))
        cv.imshow('Web Cam 3', frame_p) #Opens a window to represent the steps of the resizing
        frame_p = cv.resize(frame_p, (625, 1))
        print(np.dot((1 - (frame_p/255)), wd))
        if np.dot((1 - (frame_p/255)), wd) > 0.5: #Checking if the Face is smiling.
            cv.putText(frame, 'Smiling', (frame.shape[1]//2, frame.shape[0]), cv.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)
        else:
            cv.putText(frame, 'Not', (frame.shape[1]//2, frame.shape[0]), cv.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)

    cv.imshow('WebCam', frame) #Opens a window for the final face cam video.
    if cv.waitKey(20) & 0xFF==ord('d'): #Finishes the code when you press 'd'
        break 


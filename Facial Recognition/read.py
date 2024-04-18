import cv2 as cv
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    capture.set(3,width)
    capture.set(4,width)


img = rescaleFrame(cv.imread('Photos/Ari.png'), 3)

cv.imshow('Ari', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

shaped = cv.resize(gray, (24,24))
cv.imshow('24', shaped)

blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

'''canny1 = cv.Canny(img, 100,175)
#cv.imshow('Canny Edges', canny1)

dilate = cv.dilate(canny1, (3,3), iterations=1)
dilate2 = cv.dilate(canny1, (1001,1001), iterations=1)
#cv.imshow('dilate', dilate)
#cv.imshow('dilate1', dilate2)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.resize(gray, (80,80))
cv.imshow('Gray', gray)
print(1 - (gray/255))'''

cv.waitKey(0)
'''blank = np.zeros((500, 500, 3), dtype = 'uint8')
cv.imshow('Blank', blank)

blank[:] = 0,255,0
cv.imshow('Green',blank)

cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0, 0, 255), cv.FILLED)
cv.imshow('Rectangle', blank)
#img = cv.imread('Photos/Ari.png')
#cv.imshow('Ari', rescaleFrame(img, 2))

cv.putText(blank, 'Hello', (blank.shape[1]//2,blank.shape[0]//2), cv.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)
cv.imshow('Text', blank)'''

'''capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    
    #changeRes(320, 100)
    cv.putText(frame, 'Hello', (frame.shape[1]//2,frame.shape[0]//2), cv.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)
    #cv.imshow('WebCam Sized', frame_r)
    cv.imshow('WebCam', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

cv.destroyAllWindows()'''
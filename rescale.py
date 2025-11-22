import cv2 as cv

video = cv.VideoCapture('video.mp4')

def rescaleFrame(frame,scale=0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA )

def changeRes(width,height):
    # = for live video only 
    capture.set(3,width)
    capture.set(4,height)
     
    
while True:
    isTrue,frame= video.read()
    frame_resized = rescaleFrame(frame)
    cv.imshow('resizedvideo',frame_resized)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
video.release()
cv.destroyAllWindows()

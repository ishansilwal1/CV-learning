import cv2 as cv
image =cv.imread('image2.jpg')
# cv.imshow('check',image)
# cv.waitKey(0)

def rescaleFrame(frame,scale=0.75):
    width =int( frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA )
resized_image = rescaleFrame(image)
cv.imshow('Resized Image',resized_image)
cv.waitKey(0)

video = cv.VideoCapture('video.mp4')
while True:
    isTrue,frame = video.read()
    frame_resized=rescaleFrame(frame)
    cv.imshow('Video-resized',frame_resized)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
video.release()
cv.destroyAllWindows()



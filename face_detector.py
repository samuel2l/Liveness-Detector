# # from cvzone.FaceDetectionModule import FaceDetector
# # import cv2

# # cap=cv2.VideoCapture(0)
# # detector=FaceDetector()

# # while True:
# #     ret,frame=cap.read()
# #     img,bboxes=detector.findFaces(frame)

# #     if bboxes:
# #         #bbox format: BOX [{'id': 0, 'bbox': (491, 208, 296, 296), 'score': [0.8937399387359619], 'center': (639, 356)}]
# #         center=bboxes[0]["center"]
# #         cv2.circle(img,center,5,(255,0,255),cv2.FILLED)
# #     cv2.imshow("FRAME",frame)
# #     cv2.waitKey(1)

# # add padding to the detected face as rn it does not cover the whole face
from time import time
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2


classID = 0  # 0 is fake and 1 is real
outputFolderPath = 'dataset/fake'
save = True
debug = False
floatingPoint = 6
width_offset=0.1
height_offset=0.2

cap=cv2.VideoCapture(0)

detector=FaceDetector()

while True:
    ret,frame=cap.read()
    
    img,bboxes=detector.findFaces(frame,draw=False)
    imgOut = img.copy()
    isFaceBlur = []  
    listInfo = []  # The normalized values and the class name for the label txt file

    if bboxes:
        #bbox format: BOX [{'id': 0, 'bbox': (491, 208, 296, 296), 'score': [0.8937399387359619], 'center': (639, 356)}]
        for bbox in bboxes:
            x,y,w,h=bbox["bbox"]
            confidence = bbox["score"][0]
            if confidence>0.8:
                # note x and y are coordinates of top left corner of bbox
                off_w=width_offset*w
        #         When you compute the horizontal offset (off_w) based on the bounding box width, you want to:
        # 1.	Expand the box leftward (toward smaller x values).
        # 2.	Expand the box rightward (by increasing the width w).
                x=int(x-off_w)
                w=int(w+off_w*2)

    # In images, the coordinate system starts at the top-left corner:
    # 	•	Moving downward increases y.
    # 	•	Moving upward decreases y.
    # Exactly! The y in image coordinates behaves differently than in a standard graph:

    # Image Coordinate System:
    # 	•	The origin (0, 0) is at the top-left corner of the image.
    # 	•	x increases as you move right.
    # 	•	y increases as you move down.
                off_h=height_offset*h
                y=int(x-off_h*3)
                h=int(h+off_h*3.5)

                #in case bbox goes to a neg val like out of camera range just set it to 0 so the app does not crash
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)

# 1. cv2.Laplacian(imgFace, cv2.CV_64F)
# 	•	The Laplacian operator is a second-order derivative filter that highlights regions in the image where the intensity changes rapidly (like edges). It’s commonly used to detect edges in an image.
# 	•	The parameter cv2.CV_64F ensures the output is in a higher precision floating-point format, which is useful for later calculations.

# 2. .var()
# 	•	Once the Laplacian is applied, .var() calculates the variance of the pixel intensities in the resulting image.
# 	•	Variance measures the spread or dispersion of values. A higher variance indicates a sharper image with more details (like edges). A lower variance means the image is likely blurry because it lacks high-frequency details.

# 3. int()
# 	•	The int() function converts the variance value (a float) into an integer for easier handling.

# What Does This Achieve?
# 	•	Sharp images (non-blurry) typically have high variance in their Laplacian because there are a lot of edges.
# 	•	Blurry images have low variance because edges are less distinct.
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue >= 100:#change to 500 when doling for real and using phone
                    
                    isFaceBlur.append(True)
                else:
                    print('BLUR VALUE',blurValue)
                    isFaceBlur.append(False)

                ih, iw, _ = img.shape
                #CENTRE OF IMG
                xc, yc = x + w / 2, y + h / 2
                #normalised values
                xcn, ycn = round(xc / iw, 6), round(yc / ih, 6)
                wn, hn = round(w / iw, 6), round(h / ih, 6)
                # print(xcn, ycn, wn, hn)

# normalisez vals cannot be above 1
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                #whiole point of finding ycn den tins is cos of format yolo requires
                #format below we save our labels is format yolo requires

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ------  Drawing --------
                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(confidence * 100)}% Blur: {blurValue}', (x, y - 20),
                                   scale=2, thickness=3)
                # if debug:
                #     cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                #     cvzone.putTextRect(img, f'Score: {int(confidence * 100)}% Blur: {blurValue}', (x, y - 20),
                #                        scale=2, thickness=3)

                cv2.rectangle(imgOut,(x,y,w,h),(255,0,0),3)

        # if save:
        #     if all(not blur for blur in isFaceBlur) and isFaceBlur != []:
        #         timeNow = time()
        #         timeNow = str(timeNow).split('.')
        #         timeNow = timeNow[0] + timeNow[1]
        #         cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
        #         for info in listInfo:
        #             f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
        #             f.write(info)
        #             f.close()


    cv2.imshow("FRAME",imgOut)
    cv2.waitKey(1)


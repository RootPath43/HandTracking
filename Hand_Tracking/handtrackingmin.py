import cv2
import mediapipe as mp
import time
capture=cv2.VideoCapture(0)
mphands=mp.solutions.hands
hands=mphands.Hands()
mpDraw=mp.solutions.drawing_utils
fingerendpoints=[0,4,8,16,20]
cTime=0
pTime=0
while True:
    success, img=capture.read()
    imageHeight, imageWidth, _ = img.shape
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    #print(mphands.HandLandmark)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for points in fingerendpoints:
                print(mphands.HandLandmark(points))
                normalizedLandmark = handLms.landmark[mphands.HandLandmark(points)]
                pixelCoordinatesLandmark = mpDraw._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                          normalizedLandmark.y,
                                                                                          imageWidth, imageHeight)
                print(normalizedLandmark)
            #print(point)#fingernameandpart
            #print(pixelCoordinatesLandmark)



            mpDraw.draw_landmarks(img, handLms)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, str(int(fps)), (18,70),cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,0),3)
    cv2.imshow("img",img)
    cv2.waitKey(1)
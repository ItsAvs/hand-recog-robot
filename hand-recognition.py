import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_utils
mphands=mp.solutions.hands

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
hands = mphands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
tipIds = [4,8,12,16,20]
barIds = [2,6,10,14,18]

while True:
    data, img = webcam.read()
    img = cv2.flip(img,1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=hands.process(imgrgb)
    
    thresh = []
    count_up = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h,w,c = img.shape
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for id, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x*w), int(lm.y*h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(cx,cy)

                if id in barIds:
                    thresh.append(cy)
                if id in tipIds:
                    if cy < thresh[0]:
                        count_up += 1
                        thresh = thresh[1:]
                if id==0:
                    cv2.circle(img, (cx,cy),20,(255,0,255),cv2.FILLED)
                    #print("BOTTOM: ",cx-960,(cy-540)*-1)
                    x1, y1 = cx, cy

                if id==9:
                    cv2.circle(img, (cx,cy), 20, (255,0,255), cv2.FILLED)
                    #print("TOP: ", cx-960, (cy-540)*-1)
                    x2, y2 = cx, cy
                    px, py = round((x1+x2)/2), round((y1+y2)/2)
                    print("PALM: ", px-960,(py-540)*-1)
                    cv2.circle(img, (px,py), 20, (0,255,255), cv2.FILLED)
                    text = "("+ str(px-960) + ", " + str((py-540)*-1) +")"
                    cv2.putText(img, text, (px+10, py+10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_4)
                    coord = []
                if count_up > 3:
                    name = "PALM OPEN"
                else:
                    name = "CLOSED FIST"

                mp_drawing.draw_landmarks(img, hand_landmarks, mphands.HAND_CONNECTIONS)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            gesture_text = "GESTURE: "+name
            cv2.putText(img, gesture_text, (x_max+10, y_max+10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2, cv2.LINE_4)
            
    #cooRds are 1920 & 1080 over == (0,0)
    cv2.imshow("HANDTRACKER",img)
    cv2.waitKey(1)

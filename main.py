import cv2
import numpy as np
from hand_tracking import handDetector
import math

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    xp, yp = 0, 0
    canvas = None

    
    colors = [
    (139, 0, 0),
    (0, 100, 0),
    (0, 0, 139),
    (0, 0, 0)
]
    colorIndex = 0
    brushThickness = 7

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        if canvas is None:
            canvas = np.zeros_like(img)

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            fingers = detector.fingersUp(lmList)

            x1, y1 = lmList[8][1], lmList[8][2]

            x2, y2 = lmList[4][1], lmList[4][2]  # Thumb tip

            distance = math.hypot(x2 - x1, y2 - y1)

            brushThickness = int(np.interp(distance, [20, 200], [5, 40]))

            if fingers[1] == 1 and fingers[2] == 1:
                xp, yp = 0, 0

                if y1 < 80:
                    if 50 < x1 < 150:
                        colorIndex = 0
                    elif 200 < x1 < 300:
                        colorIndex = 1
                    elif 350 < x1 < 450:
                        colorIndex = 2
                    elif 500 < x1 < 600:
                        colorIndex = 3

            elif fingers[1] == 1 and fingers[2] == 0:
                cv2.circle(img, (x1, y1), 10, colors[colorIndex], cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                cv2.line(canvas, (xp, yp), (x1, y1),
                         colors[colorIndex], brushThickness)

                xp, yp = x1, y1

        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, canvas)

        cv2.rectangle(img, (50, 0), (150, 80), (255, 0, 0), -1)
        cv2.rectangle(img, (200, 0), (300, 80), (0, 255, 0), -1)
        cv2.rectangle(img, (350, 0), (450, 80), (0, 0, 255), -1)
        cv2.rectangle(img, (500, 0), (600, 80), (0, 0, 0), -1)

        cv2.imshow("Air Canvas", img)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('c'):
            canvas = np.zeros_like(img)

        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
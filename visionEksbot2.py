import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #masking
    #mask = cv.inRange(hsv, (90, 100, 100), (130, 255, 255)) #biru
    #mask = cv.inRange(hsv, (35, 100, 100), (85, 255, 255)) #hijau
    mask = cv.inRange(hsv, (0, 100, 100), (25, 255, 255)) #merah

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        #print(x, y, w, h)
        print(f'x={x}, y={y}, w={w}, h={h}, area={cv.contourArea(c)}')
        #cetak titik di contour
        M = cv.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        #cetak garis di contour
        cv.line(frame, (cx, cy), (640//2, 480//2), (0, 255, 0), 2)
        #jarak
        distance = np.sqrt((cx - 640//2)**2 + (cy - 480//2)**2)
        print(f'distance={distance}')
        #koreksi
        cX = cx - 320
        cY = cy - 240
        #sudut
        angle = np.arctan2(cY, cX)
        print(f'angle={angle}')
        #koreksi sudut
        angle = np.degrees(angle)
        if angle < 0:
            angle += 360
        print(f'angle={angle}')
        
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
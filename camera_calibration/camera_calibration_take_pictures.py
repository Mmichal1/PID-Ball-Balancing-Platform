import cv2 as cv
from picamera2 import Picamera2
import time

def main(args=None):
    camera = Picamera2()
    camera.preview_configuration.main.size = (640, 480)
    camera.preview_configuration.main.format = "RGB888"
    camera.preview_configuration.align()
    camera.configure("preview")
    camera.start()
    counter = 0 
    time.sleep(0.1)

    while True:
        frame = camera.capture_array()
        cv.imshow("Camera", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            cv.imwrite(f"camera_{counter}.jpeg", frame)
            counter += 1

    cv.destroyAllWindows()

if __name__=='__main__':
    main()
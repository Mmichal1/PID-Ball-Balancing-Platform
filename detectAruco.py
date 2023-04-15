from picamera2 import Picamera2
import time
import cv2 as cv

def detectAndMarkAruco(frame, arucoDetector):
    markerCorners, markerIds, rejectedCandidates = arucoDetector.detectMarkers(frame)

    if len(markerCorners) > 0:
        markerIds = markerIds.flatten()
        for (markerCorner, markerID) in zip(markerCorners, markerIds):
            (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            cv.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            cv.putText(frame, f'Twoja stara', (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(frame, str(
                markerID), (topLeft[0], topLeft[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def main(args=None):
    arucoDictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    arucoParameters = cv.aruco.DetectorParameters()
    arucoDetector = cv.aruco.ArucoDetector(arucoDictionary, arucoParameters)

    camera = Picamera2()
    camera.preview_configuration.main.size = (640, 480)
    camera.preview_configuration.main.format = "RGB888"
    camera.preview_configuration.align()
    camera.configure("preview")
    camera.start()

    time.sleep(0.1)

    while True:
        frame = detectAndMarkAruco(camera.capture_array(), arucoDetector)
        cv.imshow("Camera", frame)
        if cv.waitKey(1) == ord("q"):
            break

    cv.destroyAllWindows()

if __name__=='__main__':
    main()
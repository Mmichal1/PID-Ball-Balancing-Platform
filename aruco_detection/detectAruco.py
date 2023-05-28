from picamera2 import Picamera2
import time
import cv2 as cv
import json
import numpy as np
from datetime import datetime


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
            cv.putText(frame, str(
                markerID), (topLeft[0], topLeft[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def poseEstimationAruco(frame, matrixCoeff, distortionCoeff, arucoDetector):

    # kernel = np.ones((5,5),np.float32)/25
    # frame = cv.filter2D(frame,-1,kernel)
    frame = cv.bilateralFilter(frame,9,100,100)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    markerCorners, markerIDs, rejectedCandidates = arucoDetector.detectMarkers(gray)
    markerSizeInM = 0.057

    if len(markerCorners) > 0:
        # for i in range(0, len(markerIDs)):

            # rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(markerCorners[i], markerSizeInM, matrixCoeff, distortionCoeff)
            # cv.drawFrameAxes(frame, matrixCoeff, distortionCoeff, rvec, tvec, markerSizeInM)

        # cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIDs)

        
        markerIDs = markerIDs.flatten()
        # print(markerCorners)
        
        rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners, markerSizeInM, matrixCoeff, distortionCoeff)

        markersCenter = []

        for (markerCorner, markerID, i) in zip(markerCorners, markerIDs, range(0, markerIDs.size)):
            cv.drawFrameAxes(frame, matrixCoeff, distortionCoeff, rvec[i], tvec[i], markerSizeInM)
            # print(tvec[i])
            (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            # print(topRight)
            cv.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            markersCenter.append((cX, cY))
            distance = np.sqrt(tvec[i][0][2] ** 2 + tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2)
            cv.putText(frame, f'{distance:.2f}m', (topLeft[0], topLeft[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # print(markersCenter[0][0])
        if len(markerCorners) > 1:
            for i in range(0, len(markerIDs) - 1):
                cv.line(frame, markersCenter[i], markersCenter[i+1], (0, 255, 0), 2)
                distance = np.linalg.norm(tvec[i]-tvec[i+1])
                middle = (int((markersCenter[i][0] + markersCenter[i + 1][0]) /2), int((markersCenter[i][1] + markersCenter[i + 1][1]) /2))
                # print(middle)

                cv.circle(frame, middle, 4, (0, 0, 255), -1)
                cv.putText(frame, f'{distance:.2f}m', (middle[0] + 5, middle[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
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

    with open('../camera_calibration/camera.json', 'r') as json_file:
        camera_data = json.load(json_file)
        dist = np.array(camera_data["dist"])
        mtx = np.array(camera_data["mtx"])

    time.sleep(0.1)

    while True:
        # frame = detectAndMarkAruco(camera.capture_array(), arucoDetector)
        frame = poseEstimationAruco(camera.capture_array(), mtx, dist, arucoDetector)
        cv.imshow("Camera", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            cv.imwrite(f"output_images/camera_{datetime.now().strftime('%H%M%S')}.jpeg", frame)

    cv.destroyAllWindows()

if __name__=='__main__':
    main()
from picamera2 import Picamera2
import time
import cv2 as cv
import json
import numpy as np
from datetime import datetime


class Point:
    def __init__(self, distance: float, coordinates: 'tuple[int, int]'):    
        self.distance = distance
        self.coordinates = coordinates

def detect_markers(frame, aruco_detector):
    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(frame)

    if len(marker_corners) > 0:
        marker_ids = marker_ids.flatten()
        for (marker_corner, marker_id) in zip(marker_corners, marker_ids):
            (top_left, top_right, bottom_right, bottom_left) = marker_corner.reshape((4, 2))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            cv.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv.line(frame, bottom_left, top_left, (0, 255, 0), 2)
            cX = int((top_left[0] + bottom_right[0]) / 2.0)
            cY = int((top_left[1] + bottom_right[1]) / 2.0)
            cv.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            cv.putText(frame, str(
                marker_id), (top_left[0], top_left[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def estimate_aruco_pose(frame, matrixCoeff, distortionCoeff, aruco_detector):

    list_of_points = []

    frame = cv.bilateralFilter(frame,9,100,100)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
    markerSizeInM = 0.057

    if len(marker_corners) > 0:
        marker_ids = marker_ids.flatten()
        
        rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, markerSizeInM, matrixCoeff, distortionCoeff)

        markers_center = []

        for (marker_corner, _, i) in zip(marker_corners, marker_ids, range(0, marker_ids.size)):
            cv.drawFrameAxes(frame, matrixCoeff, distortionCoeff, rvec[i], tvec[i], markerSizeInM)
            (top_left, top_right, bottom_right, bottom_left) = marker_corner.reshape((4, 2))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            cv.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv.line(frame, bottom_left, top_left, (0, 255, 0), 2)
            cX = int((top_left[0] + bottom_right[0]) / 2.0)
            cY = int((top_left[1] + bottom_right[1]) / 2.0)
            markers_center.append((cX, cY))
            distance = np.sqrt(tvec[i][0][2] ** 2 + tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2)
            cv.putText(frame, f'{distance:.2f}m', (top_left[0], top_left[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if len(marker_corners) > 1:
            for i in range(len(marker_ids)):
                for j in range(i + 1, len(marker_ids)):
                    distance = np.linalg.norm(tvec[i]-tvec[j])
                    middle = (int((markers_center[i][0] + markers_center[j][0]) / 2), int((markers_center[i][1] + markers_center[j][1]) / 2))
                    list_of_points.append(Point(distance=distance, coordinates=(middle)))

        if (len(list_of_points) > 0):            
            sorted_points = sorted(list_of_points, key=lambda p: p.distance, reverse=True)
            print(f'{sorted_points[0].distance} {sorted_points[0].coordinates}')
            cv.circle(frame, sorted_points[0].coordinates, 10, (0, 0, 255), -1)
        
    return frame


def detect_ball(frame, known_ball_size, camera_focal_len):
    # Convert the image to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the ball color (example values)
    lower_color = np.array([0, 150, 190])
    upper_color = np.array([180, 255, 255])

    # Create a mask for the ball color
    mask = cv.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to remove noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Find contours of the ball
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the ball)
    if len(contours) > 0:
        ball_contour = max(contours, key=cv.contourArea)
        (x, y), radius = cv.minEnclosingCircle(ball_contour)
        # return (int(x), int(y)), int(radius)
        
        ball_size = 2 * int(radius)

        height, width, _ = frame.shape

        # Transform the x and y coordinates
        x_transformed = x - width / 2
        y_transformed = height / 2 - y
        

        # Estimate distance
        distance = (known_ball_size * camera_focal_len) / ball_size
    
        # Convert distance to translation vector
        print(x_transformed, y_transformed, distance)
        cv.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    return frame

def main(args=None):
    known_ball_size = 0.065

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
        mtx = np.array(camera_data["mtx"]) # camera focal length is mtx[0][0]

    time.sleep(0.1)

    while True:
        # frame = detect_markers(camera.capture_array(), aruco_detector)
        frame = estimate_aruco_pose(camera.capture_array(), mtx, dist, arucoDetector)
        # frame = detect_ball(camera.capture_array(), known_ball_size, int(mtx[0][0]))
        cv.imshow("Camera", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            cv.imwrite(f"output_images/camera_{datetime.now().strftime('%H%M%S')}.jpeg", frame)

    cv.destroyAllWindows()

if __name__=='__main__':
    main()

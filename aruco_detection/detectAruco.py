from picamera2 import Picamera2
import time
import cv2 as cv
import json
import numpy as np
import math
from datetime import datetime

"""
 - Find the ranges of possible center positions 
 Steps of finding the center of the platform:
 - Check if the calculated point is within possible boundaries 
 - If yes then save that point as the middle 
 - If not then discard it and used last saved point
"""

platform_middle = (300, 230)
previous_platform_middle = (300, 230)
x_boundaries = (280, 330)
y_boundaries = (200, 260)

class Segment:
    def __init__(self, length: float, start_point: 'tuple[int, int]', end_point: 'tuple[int, int]'):    
        self.start_point = start_point
        self.end_point = end_point
        self.length = length

def estimate_aruco_pose(frame, matrix_coeff, distortion_coeff, aruco_detector):

    known_ball_size = 0.038

    ball_coordinates = detect_ball(frame, known_ball_size, int(matrix_coeff[0][0]))
    
    list_of_segments = []

    frame = cv.bilateralFilter(frame,9,100,100)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
    marker_size_m = 0.047

    if len(marker_corners) > 0:
        global platform_middle
        global previous_platform_middle
        markers_center = []
        marker_ids = marker_ids.flatten()
        rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, marker_size_m, matrix_coeff, distortion_coeff)

        for (marker_corner, _, i) in zip(marker_corners, marker_ids, range(0, marker_ids.size)):
            cv.drawFrameAxes(frame, matrix_coeff, distortion_coeff, rvec[i], tvec[i], marker_size_m)
            (top_left, top_right, bottom_right, bottom_left) = marker_corner.reshape((4, 2))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            cX = int((top_left[0] + bottom_right[0]) / 2.0)
            cY = int((top_left[1] + bottom_right[1]) / 2.0)
            markers_center.append((cX, cY))
            distance = np.sqrt(tvec[i][0][2] ** 2 + tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2)
            cv.putText(frame, f'{distance:.2f}m', (top_left[0], top_left[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if len(marker_corners) > 1:
            for i in range(len(marker_ids)):
                for j in range(i + 1, len(marker_ids)):
                    
                    distance = np.linalg.norm(tvec[i]-tvec[j])
                    cv.line(frame, markers_center[i], markers_center[j], (0, 255, 0), 2)
                    list_of_segments.append(Segment(length=distance, start_point=(markers_center[i]), end_point=(markers_center[j])))
        
        if len(list_of_segments) > 1:            
            
            sorted_segments = sorted(list_of_segments, key=lambda p: p.length, reverse=True)
            platform_middle = find_intersection(sorted_segments[0], sorted_segments[1])
            
            if not (platform_middle[0] > x_boundaries[0] and platform_middle[0] < x_boundaries[1] and platform_middle[1] > y_boundaries[0] and platform_middle[1] < y_boundaries[1]):
                platform_middle = previous_platform_middle
                
            previous_platform_middle = platform_middle
            
        else: 
            platform_middle = previous_platform_middle
            
        cv.circle(frame, platform_middle, 10, (0, 0, 255), -1)
        #print(platform_middle)
            
        if ball_coordinates is not None:
            cv.line(frame, platform_middle, ball_coordinates, (0, 255, 0), 2)
            result = tuple(x - y for x, y in zip(platform_middle, ball_coordinates))
            # print(result)
            cv.circle(frame, ball_coordinates, 4, (0, 0, 255), -1)
            

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

        # Estimate distance
        distance = (known_ball_size * camera_focal_len) / ball_size
    
        # Convert distance to translation vector
        # print(int(x), int(y), distance)

        return (int(x), int(y))

    return None

def find_intersection(segment_one, segment_two):
    x1, y1 = segment_one.start_point
    x2, y2 = segment_one.end_point
    x3, y3 = segment_two.start_point
    x4, y4 = segment_two.end_point
    
    # Calculate the intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    
    return int(px), int(py)
    

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
        mtx = np.array(camera_data["mtx"]) # camera focal length is mtx[0][0]

    time.sleep(0.1)

    while True:
        frame = estimate_aruco_pose(camera.capture_array(), mtx, dist, arucoDetector)
        cv.imshow("Camera", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            cv.imwrite(f"output_images/camera_{datetime.now().strftime('%H%M%S')}.jpeg", frame)

    cv.destroyAllWindows()

if __name__=='__main__':
    main()

from picamera2 import Picamera2
import time
import cv2 as cv
import json
import numpy as np
import math
import serial
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
ball_x_boundaries = (-220, 220)
ball_y_boundaries = (-220, 220)
#x_boundaries = (-4.0, 4.0)
#y_boundaries = (-4.0, 4.0)
#ball_x_boundaries = (-200.0, 200.0)
#ball_y_boundaries = (-200.0, 200.0)
center_tvec = [0, 0, 0]

# Ustalenie wartości początkowych
y_distance_previous_error = 0.0
y_distance_error = 0.0
x_distance_previous_error = 0.0
x_distance_error = 0.0
period = 0.05  # Czas odświeżania pętli w sekundach (50 ms)

'''
kp = 0.3
ki = 0.02
kd = 0.15
'''

# Ustalenie wartości stałych regulatora PID
kp = 0.38
ki = 0.022
kd = 0.1
distance_setpoint = 0.0
y_PID_p = 0.0
y_PID_i = 0.0
y_PID_d = 0.0
y_PID_total = 0.0
x_PID_p = 0.0
x_PID_i = 0.0
x_PID_d = 0.0
x_PID_total = 0.0

servo_angle = 60  # Wyjściowy sygnał dla położenia neutralnego

class Segment:
    def __init__(self, length: float, start_point: 'tuple[int, int]', end_point: 'tuple[int, int]', start_tvec: 'list[float, float, float]', end_tvec: 'list[float, float, float]'):    
        self.start_point = start_point
        self.end_point = end_point
        self.length = length
        self.start_tvec = start_tvec
        self.end_tvec = end_tvec

def estimate_aruco_pose(frame, matrix_coeff, distortion_coeff, aruco_detector):

    known_ball_size = 3.8

    ball_coordinates, ball_tvec = detect_ball(frame, known_ball_size, int(matrix_coeff[0][0]))
    
    list_of_segments = []

    frame = cv.bilateralFilter(frame,9,100,100)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
    marker_size_cm = 4.7

    if len(marker_corners) > 0:
        global platform_middle
        global previous_platform_middle
        global center_tvec
        markers_center = []
        marker_ids = marker_ids.flatten()
        rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, marker_size_cm, matrix_coeff, distortion_coeff)

        for (marker_corner, _, i) in zip(marker_corners, marker_ids, range(0, marker_ids.size)):
            #print(tvec[i])
            cv.drawFrameAxes(frame, matrix_coeff, distortion_coeff, rvec[i], tvec[i], marker_size_cm)
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
                    list_of_segments.append(Segment(length=distance, start_point=(markers_center[i]), end_point=(markers_center[j]), start_tvec=(tvec[i]), end_tvec=(tvec[j])))
        
        if len(list_of_segments) > 1:            
            
            sorted_segments = sorted(list_of_segments, key=lambda p: p.length, reverse=True)
            platform_middle, center_tvec = find_intersection(sorted_segments[0], sorted_segments[1])
            
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
            #result_tvec = [x - y for x, y in zip(center_tvec, ball_tvec)]
            #print(f'result: {result_tvec}\ncenter: {center_tvec}\nball: {ball_tvec}')
            cv.circle(frame, ball_coordinates, 4, (0, 0, 255), -1)
            
            if (result[0] > ball_x_boundaries[0] and result[0] < ball_x_boundaries[1] and result[1] > ball_y_boundaries[0] and result[1] < ball_y_boundaries[1]):
                return frame, result
            
            return frame, (0, 0)
            

    return frame, (0, 0)


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
    
        height, width = frame.shape[:2]
        
        transformed_x = x - (width / 2)
        transformed_y = (height / 2) - y
        
        ball_tvec = [transformed_x, transformed_x, distance]
    
        # Convert distance to translation vector
        #print(ball_tvec)
        return (int(x), int(y)), ball_tvec
        #return (ball_tvec)

    return None, None

def find_intersection(segment_one, segment_two):
    x1, y1 = segment_one.start_point
    x2, y2 = segment_one.end_point
    x3, y3 = segment_two.start_point
    x4, y4 = segment_two.end_point
    
    # Calculate the intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    
    middle = (segment_one.start_tvec[0] + segment_one.end_tvec[0]) / 2
    #print(segment_one.start_tvec[0])
    #print(middle)
    
    return (int(px), int(py)), middle
     
def map_value(value, in_min, in_max, out_min, out_max):
    # Map a value from one range to another range
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
     
def PID_y(distance):
    global y_distance_previous_error
    global y_distance_error
    global period
    global kp 
    global ki
    global kd
    global distance_setpoint
    global y_PID_p
    global y_PID_i
    global y_PID_d
    global y_PID_total
    
    y_distance_error = distance_setpoint - distance
    y_PID_p = kp * y_distance_error
    y_PID_d = kd * ((y_distance_error - y_distance_previous_error) / period)

    if -40 < y_distance_error < 40:
        y_PID_i += ki * y_distance_error
    else:
        y_PID_i = 0

    y_PID_total = y_PID_p + y_PID_i + y_PID_d
    print(f'y_PID_p: {y_PID_p},\npid_i: {y_PID_i},\npid_d: {y_PID_d},')
    #print(y_PID_d)
    #y_PID_total = (y_PID_total + 70) * 0.85
    y_PID_total = map_value(y_PID_total, -190, 190, 41, 78)
    


    if y_PID_total < 41:
        y_PID_total = 41
    if y_PID_total > 78:
        y_PID_total = 78

    y_distance_previous_error = y_distance_error
        
    print (f'Y TOTAL:{int(y_PID_total)}')
    
    return int(y_PID_total)


def PID_x(distance):
    global x_distance_previous_error
    global x_distance_error
    global period
    global kp
    global ki
    global kd
    global distance_setpoint
    global x_PID_p
    global x_PID_i
    global x_PID_d
    global x_PID_total

    x_distance_error = distance_setpoint - distance
    x_PID_p = kp * x_distance_error
    x_PID_d = kd * ((x_distance_error - x_distance_previous_error) / period)

    if -40 < x_distance_error < 40:
        x_PID_i += ki * x_distance_error
    else:
        x_PID_i = 0

    x_PID_total = x_PID_p + x_PID_i + x_PID_d
    print(f'x_PID_p: {x_PID_p},\npid_i: {x_PID_i},\npid_d: {x_PID_d},')
    # print(x_PID_d)
    # x_PID_total = (x_PID_total + 70) * 0.85
    x_PID_total = map_value(x_PID_total, -190, 190, 51, 86)

    if x_PID_total < 51:
        x_PID_total = 51
    if x_PID_total > 86:
        x_PID_total = 86

    x_distance_previous_error = x_distance_error

    print (f'X TOTAL:{int(x_PID_total)}')

    return int(x_PID_total)

def main(args=None):
    serial_port = serial.Serial('/dev/ttyACM0', 74880, timeout=1)
    serial_port.reset_input_buffer()
    serial_port.write(b"60,65\n")

    arucoDictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    arucoParameters = cv.aruco.DetectorParameters()
    arucoDetector = cv.aruco.ArucoDetector(arucoDictionary, arucoParameters)

    camera = Picamera2()
    camera.preview_configuration.main.size = (640, 480)
    #camera.preview_configuration.main.size = (1296, 972)
    camera.preview_configuration.main.format = "RGB888"
    camera.preview_configuration.align()
    camera.configure("preview")
    camera.start()

    with open('../camera_calibration/camera.json', 'r') as json_file:
        camera_data = json.load(json_file)
        dist = np.array(camera_data["dist"])
        mtx = np.array(camera_data["mtx"]) # camera focal length is mtx[0][0]

    time.sleep(0.01)

    while True:
        #line = serial_port.readline().decode('utf-8').rstrip()
        #print(line)
        
        frame, result = estimate_aruco_pose(camera.capture_array(), mtx, dist, arucoDetector)
        print(result)
        
        servo_value_y = PID_y(-result[0])
        #servo_value_y = 58
        servo_value_x = PID_x(result[1])
        #servo_value_x = 67
        
        serial_port.write(b"%d,%d\n" % (servo_value_y, servo_value_x))
        # serial_port.write(b"60,65\n")
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        cv.imshow("Camera", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            cv.imwrite(f"output_images/camera_{datetime.now().strftime('%H%M%S')}.jpeg", frame)
            
        time.sleep(period)

    cv.destroyAllWindows()

if __name__=='__main__':
    main()

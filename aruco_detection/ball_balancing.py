from picamera2 import Picamera2
import time
import cv2 as cv
import json
import numpy as np
import serial
from my_pid import MyPID
from segment import Segment
from datetime import datetime

# Initial platform center position 
platform_center = (300, 230)
previous_platform_center = (300, 230)

# Platform center boundaries
x_boundaries = (280, 330)
y_boundaries = (200, 260)

# Ball position boundaries
ball_x_boundaries = (-220, 220)
ball_y_boundaries = (-220, 220)

# Loop refresh rate
period = 0.05  

def get_ball_dist_from_center(frame, matrix_coeff, distortion_coeff, aruco_detector):
    # Constants
    known_ball_size = 3.8
    marker_size_cm = 4.7

    list_of_segments = []

    # This filter blurs all the details but leaves the contours
    # so that processing the frame to find markers is easier and 
    # more accurate
    frame = cv.bilateralFilter(frame, 9, 100, 100)

    # Process the frame, get the ball position and mark it on the frame 
    ball_coordinates = detect_ball(frame, known_ball_size, int(matrix_coeff[0][0]))

    # Make the frame grayscale as the program doesn't need RGB values
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect markers in the freme 
    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)

    if len(marker_corners) > 0:
        global platform_center
        global previous_platform_center
        markers_center = []
        marker_ids = marker_ids.flatten()

        # Estimate pose of the markers in relation to the camera position
        rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, marker_size_cm, matrix_coeff, distortion_coeff)

        for (marker_corner, _, i) in zip(marker_corners, marker_ids, range(0, marker_ids.size)):
            # Draw frame axes for each marker 
            cv.drawFrameAxes(frame, matrix_coeff, distortion_coeff, rvec[i], tvec[i], marker_size_cm)

            # Find and save markers center
            (top_left, top_right, bottom_right, bottom_left) = marker_corner.reshape((4, 2))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            cX = int((top_left[0] + bottom_right[0]) / 2.0)
            cY = int((top_left[1] + bottom_right[1]) / 2.0)
            markers_center.append((cX, cY))

            # Get marker distance from the camera and write in on the frame
            distance = np.sqrt(tvec[i][0][2] ** 2 + tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2)
            cv.putText(frame, f'{distance:.2f}m', (top_left[0], top_left[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Find all segments from every marker to every other one and save each segment data 
        if len(marker_corners) > 1:
            for i in range(len(marker_ids)):
                for j in range(i + 1, len(marker_ids)):
                    # Calculate the segments length
                    distance = np.linalg.norm(tvec[i]-tvec[j])
                    cv.line(frame, markers_center[i], markers_center[j], (0, 255, 0), 2)
                    list_of_segments.append(Segment(length=distance, start_point=(markers_center[i]), end_point=(markers_center[j]), start_tvec=(tvec[i]), end_tvec=(tvec[j])))
        

        if len(list_of_segments) > 1:            
            # Sort the segments by their length 
            sorted_segments = sorted(list_of_segments, key=lambda p: p.length, reverse=True)
            # The platform center will be the intersection between two longest segments which will be the diagonals
            platform_center = find_intersection(sorted_segments[0], sorted_segments[1])
            
            # Check if the platform center is within the possible coordinates if not then keep the previous calculated center as the current one
            if not (platform_center[0] > x_boundaries[0] and platform_center[0] < x_boundaries[1] and platform_center[1] > y_boundaries[0] and platform_center[1] < y_boundaries[1]):
                platform_center = previous_platform_center
                
            previous_platform_center = platform_center

        else: 
            platform_center = previous_platform_center
            
        # Mark platform center on the frame 
        cv.circle(frame, platform_center, 10, (0, 0, 255), -1)
            
        # If the ball is within the frame and platform boundaries then calculate the ball coordinates 
        if ball_coordinates is not None:
            cv.line(frame, platform_center, ball_coordinates, (0, 255, 0), 2)
            result = tuple(x - y for x, y in zip(platform_center, ball_coordinates))
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
        (x, y), _ = cv.minEnclosingCircle(ball_contour)
        return (int(x), int(y))

    return None, None

def find_intersection(segment_one, segment_two):
    # Read points coordinates
    x1, y1 = segment_one.start_point
    x2, y2 = segment_one.end_point
    x3, y3 = segment_two.start_point
    x4, y4 = segment_two.end_point
    
    # Calculate the intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    
    return (int(px), int(py))

def main(args=None):
    # PID constants
    k_p = 0.38
    k_i = 0.022
    k_d = 0.1
    setpoint = 0

    # Init serial port connection
    serial_port = serial.Serial('/dev/ttyACM0', 74880, timeout=1)
    serial_port.reset_input_buffer()
    # Send signal to level the platform
    serial_port.write(b"60,65\n")

    # Create PID instances for each axis
    PID_x = MyPID(k_p=k_p, k_i=k_i, k_d=k_d, setpoint=setpoint, servo_lower_bound=51, servo_upper_bound=80)
    PID_y = MyPID(k_p=k_p, k_i=k_i, k_d=k_d, setpoint=setpoint, servo_lower_bound=41, servo_upper_bound=78)

    # Init aruco detector 
    arucoDictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    arucoParameters = cv.aruco.DetectorParameters()
    arucoDetector = cv.aruco.ArucoDetector(arucoDictionary, arucoParameters)

    # Init and configure PiCamera 
    camera = Picamera2()
    camera.preview_configuration.main.size = (640, 480)
    camera.preview_configuration.main.format = "RGB888"
    camera.preview_configuration.align()
    camera.configure("preview")
    camera.start()

    # Read camera parameters received from camera calibration
    with open('../camera_calibration/camera.json', 'r') as json_file:
        camera_data = json.load(json_file)
        dist = np.array(camera_data["dist"])
        mtx = np.array(camera_data["mtx"])

    # Wait for camera to be ready
    time.sleep(0.01)

    # Main program loop
    while True:
        # Capture frame, process it to find markers then the platform center and ball coordinates in 
        # relation to the platform center. Mark all markers, ball position, platform center and all lines
        # on the frame
        frame, result = get_ball_dist_from_center(camera.capture_array(), mtx, dist, arucoDetector)
        print(f'Ball coordinates: {result}')
        
        # Get values that should be sent to the servos based on the distance in each axis from the center
        servo_value_y = PID_y.regulate(-result[0])
        servo_value_x = PID_x.regulate(result[1])
        serial_port.write(b"%d,%d\n" % (servo_value_y, servo_value_x))

        # Show camera feed in a separate window
        cv.imshow("Camera", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            cv.imwrite(f"output_images/camera_{datetime.now().strftime('%H%M%S')}.jpeg", frame)
        
        # Wait to enter the next iteration
        time.sleep(period)

    cv.destroyAllWindows()

if __name__=='__main__':
    main()
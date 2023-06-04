import time
import RPi.GPIO as GPIO


# Ustalenie wartości początkowych
distance_previous_error = 0.0
distance_error = 0.0
period = 0.05  # Czas odświeżania pętli w sekundach (50 ms)

# Ustalenie wartości stałych regulatora PID
kp = 8.0
ki = 0.2
kd = 3100.0
distance_setpoint = 21.0
PID_p = 0.0
PID_i = 0.0
PID_d = 0.0
PID_total = 0.0

servo_angle = 60  # Wyjściowy sygnał dla położenia neutralnego

try:
    while True:
        if time.time() > period:
            start_time = time.time()
            distance = get_dist(100)
            distance_error = distance_setpoint - distance
            PID_p = kp * distance_error
            dist_difference = distance_error - distance_previous_error
            PID_d = kd * ((distance_error - distance_previous_error) / period)

            if -3 < distance_error < 3:
                PID_i += ki * distance_error
            else:
                PID_i = 0

            PID_total = PID_p + PID_i + PID_d
            PID_total = (PID_total + 150) * 0.3

            if PID_total < 20:
                PID_total = 20
            if PID_total > 160:
                PID_total = 160

            pwm.ChangeDutyCycle(PID_total)
            distance_previous_error = distance_error

            execution_time = time.time() - start_time
            if execution_time < period:
                time.sleep(period - execution_time)
except KeyboardInterrupt:
    pass

pwm.stop()
GPIO.cleanup()

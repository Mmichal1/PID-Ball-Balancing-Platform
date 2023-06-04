import RPi.GPIO as GPIO
import time

def set_pwm_value(value):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)
    
    pwm = GPIO.PWM(18, 50)
    pwm.start(0)
    
    duty_cycle = value / 18 + 2
    pwm.ChangeDutyCycle(duty_cycle)
    
    time.sleep(0.1)  # Poczekaj 0.5 sekundy między kolejnymi wartościami
    
def move_servo_slowly(min_value, max_value, step):
    for value in range(min_value, max_value + 1, step):
        set_pwm_value(value)
    
    for value in range(max_value, min_value - 1, -step):
        set_pwm_value(value)

# Przykładowe użycie funkcji
min_value = 10  # Minimalna wartość (0-100)
max_value = 90  # Maksymalna wartość (0-100)
step = 5  # Krok zmiany wartości

move_servo_slowly(min_value, max_value, step)
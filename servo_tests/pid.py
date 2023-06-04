import time

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint

        self.last_error = 0
        self.integral = 0

    def calculate(self, feedback_value):
        error = self.setpoint - feedback_value
        self.integral += error
        derivative = error - self.last_error

        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        self.last_error = error
        return output

# Przykładowe wartości współczynników PID oraz punktu zadanego
Kp = 0.5
Ki = 0.2
Kd = 0.1
setpoint = 0

# Tworzenie obiektu kontrolera PID
pid_controller = PIDController(Kp, Ki, Kd, setpoint)

# Symulacja ruchu serwomechanizmu
feedback_value = 0  # Początkowa wartość odczytu z czujnika
target_time = time.time() + 10  # Czas symulacji - 10 sekund

while time.time() < target_time:
    output = pid_controller.calculate(feedback_value)

    # Aktualizacja stanu serwomechanizmu na podstawie wartości wyjściowej
    # np. ustawianie wychylenia rynny, sterowanie silnikiem, itp.

    # Symulacja zmiany wartości odczytu z czujnika w czasie
    feedback_value += output * 0.1  # Przykładowa zmiana wartości czujnika (0.1 to krok symulacji)
    time.sleep(0.1)  # Czas oczekiwania na kolejną iterację

print("Symulacja zakończona.")

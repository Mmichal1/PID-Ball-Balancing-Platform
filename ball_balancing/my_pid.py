class MyPID:
    def __init__(self, k_p: float, k_i: float, k_d: float, setpoint: int, period: float, servo_upper_bound: int, servo_lower_bound: int, integral_bounds: int):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.setpoint = setpoint
        self.period = period
        self.integral_bounds = integral_bounds
        self.servo_lower_bound = servo_lower_bound
        self.servo_upper_bound = servo_upper_bound
        self.PID_p = 0.0
        self.PID_i = 0.0
        self.PID_d = 0.0
        self.PID_total = 0.0
        self.distance_previous_error = 0.0
        self.distance_error = 0.0

    def _map_value(value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def regulate(self, setpoint_error: int):
        self.distance_error = self.setpoint - setpoint_error
        # Proportional
        self.PID_p = self.k_p * self.distance_error

        # Integral
        if - self.integral_bounds < self.distance_error < self.integral_bounds:
            self.PID_i += self.k_i * self.distance_error
        else:
            self.PID_i = 0

        # Derivative
        self.PID_d = self.k_d * \
            ((self.distance_error - self.distance_previous_error) / self.period)

        self.PID_total = self.PID_p + self.PID_i + self.PID_d
        self.PID_total = self._map_value(self.PID_total, -190, 190, self.servo_lower_bound, self.servo_upper_bound)

        if self.PID_total < self.servo_lower_bound:
            self.PID_total = self.servo_lower_bound
        if self.PID_total > self.servo_upper_bound:
            self.PID_total = self.servo_upper_bound

        print(
            f'Y PID_p: {self.PID_p},\nY PID_i: {self.PID_i},\nY PID_d: {self.PID_d},\nY TOTAL:{int(self.PID_total)}')

        self.distance_previous_error = self.distance_error

        return int(self.PID_total)
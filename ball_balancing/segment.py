class Segment:
    def __init__(self, length: float, start_point: 'tuple[int, int]', end_point: 'tuple[int, int]', start_tvec: 'list[float, float, float]', end_tvec: 'list[float, float, float]'):
        self.start_point = start_point
        self.end_point = end_point
        self.length = length
        self.start_tvec = start_tvec
        self.end_tvec = end_tvec
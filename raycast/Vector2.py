import math
import random


class Vector2:
    # TODO katasdk full pygame.Vector2 support
    # ghast made his own >:(

    def __init__(self, x, y=0.0):
        if isinstance(x, Vector2):
            self.x = x.x
            self.y = x.y
        else:
            self.x = x
            self.y = y

    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        else:
            return self.y

    def __len__(self):
        return 2

    def __iter__(self):
        return (v for v in (self.x, self.y))

    def __add__(self, other: 'Vector2'):
        return Vector2(self.x + other[0], self.y + other[1])

    def __sub__(self, other: 'Vector2'):
        return Vector2(self.x - other[0], self.y - other[1])

    def __mul__(self, other: float):
        return Vector2(self.x * other, self.y * other)

    def __neg__(self):
        return Vector2(-self.x, -self.y)

    def __eq__(self, other: 'Vector2'):
        return self.x == other[0] and self.y == other[1]

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return "Vector2({}, {})".format(self.x, self.y)

    def dot(self, other):
        return self.x * other[0] + self.y * other[1]

    def rotate_ip(self, degrees):
        theta = math.radians(degrees)
        cs = math.cos(theta)
        sn = math.sin(theta)
        x = self.x * cs - self.y * sn
        y = self.x * sn + self.y * cs
        self.x = x
        self.y = y

    def rotate(self, degrees):
        res = Vector2(self)
        res.rotate_ip(degrees)
        return res

    def to_ints(self):
        return Vector2(int(self.x), int(self.y))

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def length_squared(self):
        return self.x * self.x + self.y * self.y

    def scale_to_length(self, length, or_else_rand=True) -> 'Vector2':
        cur_length = self.length()
        if cur_length == 0 and length != 0:
            if or_else_rand:
                res = Vector2(0, length)
                res.rotate_ip(360 * random.random())
                return res
            else:
                raise ValueError("Cannot scale vector with length 0")
        else:
            mult = length / cur_length
            return self * mult

    def angle_to(self, other):
        dot = self.dot(other)
        mags = self.length() * math.sqrt(other[0] * other[0] + other[1] * other[1])
        if mags > 0.001:
            neg_dot = dot < 0
            dot = abs(dot)
            det = dot / mags
            if det <= 0.999:
                if neg_dot:
                    return 180 - math.degrees(math.acos(det))
                else:
                    return math.degrees(math.acos(det))
        return 0

    def distance_to(self, other):
        return (self - other).length()

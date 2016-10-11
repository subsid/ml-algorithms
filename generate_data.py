import random; 
import numpy as np;

## () -> (x_1, x_2)
def create_random_point():
    return [random.uniform(-1, 1), random.uniform(-1, 1)]

# :: () -> (Point -> Classification.Class)
#Vector Point :: (Int, Int)
# Classification.Class :: Int1Or0
def create_target_function():
    p1 = create_random_point()
    p2 = create_random_point()
    def evaluate_tf(p):
        # 2 point form of line.
        # (y - y1 / y2 - y1) - (x - x1 / x2 - x1)
        return ((p[1] - p1[1]) / (p2[1] - p1[1])) - ((p[0] - p1[0]) / (p2[0] - p1[0]))

    return lambda p: (1 if (evaluate_tf(p) >= 0) else -1)

# :: TargetFunction -> Point -> [[1, Float, Float], Classification.Class]
def create_pair(f, point):
    return [[1.0] + point, f(point)]

# :: [Point] ->  -> PointOrNone
def find_misclassified_points(points, h):
    return [p for p in points if h(p[0]) != p[1]]

def create_hypothesis_function(weights):
    return lambda p: 1 if np.dot(weights, p) >= 0 else -1


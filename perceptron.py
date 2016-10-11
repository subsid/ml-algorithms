## Perceptron algorithm
import random
import numpy as np

## () -> (x_1, x_2)
def create_random_point():
    return [random.uniform(-1, 1), random.uniform(-1, 1)]

# :: () -> (Point -> Classification.Class)
# Point :: (Int, Int)
# Classification.Class :: Int1Or0
def create_target_function():
    p1 = create_random_point()
    p2 = create_random_point()
    def evaluate_tf(p):
        # 2 point form of line.
        # (y - y1 / y2 - y1) - (x - x1 / x2 - x1)
        return ((p[1] - p1[1]) / (p2[1] - p1[1])) - ((p[0] - p1[0]) / (p2[0] - p1[0]))

    return lambda p: (1 if (evaluate_tf(p) >= 0) else -1)

# :: TargetFunction -> Point -> [TargetFunction, Classification.Class]
def create_training_pair(f, point):
    return (point, f(point))

# :: [Point] -> Numpy.Vector -> PointOrNone
def find_misclassified_points(points, h):
    return [p for p in points if h(p[0]) != p[1]]

def create_hypothesis_function(weights):
    return lambda point: 1 if np.dot(weights, np.array([1, point[0], point[1]])) >= 0 else -1

def compute_avg_error_and_iters_for_convergence(f, training_set_size=100, test_set_size=100):
    weights = np.array([0.0, 0.0, 0.0])
    h = create_hypothesis_function(weights)

    training_set = [create_training_pair(f, create_random_point()) for i in range(training_set_size)]
    test_set = [create_random_point() for i in range(test_set_size)]
    misclassified_points = find_misclassified_points(training_set, h)

    iters = 0
    while(len(misclassified_points) != 0):
        misclassified_point = random.choice(misclassified_points)
        misclassified_label = misclassified_point[1]
        weights += misclassified_label * np.array([1, misclassified_point[0][0], misclassified_point[0][1]])
        iters += 1

        updated_h = create_hypothesis_function(weights)
        misclassified_points = find_misclassified_points(training_set, updated_h)

    g = create_hypothesis_function(weights)
    test_error = sum([1.0 if g(p) != f(p) else 0 for p in test_set])/test_set_size
    return [iters, test_error]


itersAvg = []
pAvg = []

for i in range(1000):
    f = create_target_function()
    [iters, error] = compute_avg_error_and_iters_for_convergence(f)
    itersAvg.append(iters)
    pAvg.append(error)

print(sum(itersAvg)/1000.0)
print(sum(pAvg)/1000.0)


import random;
import numpy as np;

import generate_data as data;
from linear_regression import compute_avg_error_regression

iters = 1000
N = 1000
test_size = 1000
noisy_N_size = int(N/10)
e_in_linear_total = 0.0
e_out_linear_total = 0.0
e_in_non_linear_total = 0.0
e_out_non_linear_total = 0.0

# [1, x1, x2] -> [1, x1, x2, x1x2, x1^2, x2^2]
def add_non_linear_features(d):
    return d + [d[1] * d[2], d[1] ** 2, d[2] ** 2]


if __name__ == "__main__":
    for i in range(1000):
        print(i)
        f = lambda p: 1 if (p[0] ** 2 + p[1] ** 2 - 0.6) >= 0 else -1
        training_data = [data.create_pair(f, data.create_random_point()) for i in range(N)]
        test_data = [data.create_pair(f, data.create_random_point()) for i in range(test_size)]
        for _ in range(noisy_N_size):
            i = random.randint(0, N-1)
            training_data[i][1] *= -1

        for _ in range(noisy_N_size):
            i = random.randint(0, N-1)
            test_data[i][1] *= -1

        training_data_with_nonlinear_features = [[add_non_linear_features(d[0]), d[1]] for d in training_data]
        test_data_with_nonlinear_features = [[add_non_linear_features(d[0]), d[1]] for d in test_data]

        weights, e_in_linear, e_out_linear = compute_avg_error_regression(training_data, test_data)
        weights_2, e_in_non_linear, e_out_non_linear = compute_avg_error_regression(training_data_with_nonlinear_features, test_data_with_nonlinear_features)

        e_in_linear_total += e_in_linear
        e_out_linear_total += e_out_linear
        e_in_non_linear_total += e_in_non_linear
        e_out_non_linear_total += e_out_non_linear

    print("E_in_linear", e_in_linear_total/iters)
    print("E_out_linear", e_out_linear_total/iters)
    print("E_in_non_linear", e_in_non_linear_total/iters)
    print("E_out_non_linear", e_out_non_linear_total/iters)


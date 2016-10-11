import random;
import numpy as np;

import generate_data as data;

def pla(initial_weights, training_data):
    weights = initial_weights

    h = data.create_hypothesis_function(weights)
    misclassified_points = data.find_misclassified_points(training_data, h)

    iters_until_convergence = 0
    while(len(misclassified_points) != 0):
        iters_until_convergence += 1
        misclassified_point = random.choice(misclassified_points)
        misclassified_label = misclassified_point[1]
        weights += misclassified_label * np.array(misclassified_point[0])

        updated_h = data.create_hypothesis_function(weights)
        misclassified_points = data.find_misclassified_points(training_data, updated_h)

    return iters_until_convergence

def compute_avg_error_regression(training_data, test_data):
    N = len(training_data)
    test_size = len(test_data)

    y = np.array([pair[1] for pair in training_data])
    X = np.array([pair[0] for pair in training_data])
    # X_trans = np.transpose(X)
    # X_X_trans_inverse = np.linalg.inv(np.matmul(X_trans, X))

    ## (X_trans * X)^-1 * X^T 
    ## This can also be found using linalg.pinv of X
    # X_dagger = np.matmul(X_X_trans_inverse, X_trans)
    X_dagger = np.linalg.pinv(X)
    weights = np.matmul(X_dagger, y)

    g = data.create_hypothesis_function(weights)

    E_in = [1 for p in training_data if g(p[0]) != p[1]]
    E_out = [1 for p in test_data if g(p[0]) != p[1]]

    if (test_size != 0):
        return [weights, len(E_in)/float(N), len(E_out)/float(test_size)]
    else:
        return [weights, len(E_in)/float(N)]

if __name__ == "__main__":
    N = 100
    pla_N = 10
    test_size = 1000
    iters = 1000
    e_in_lr_total = 0.0
    e_out_lr_total = 0.0
    iters_total = 0

    for iters in range(iters):
        print(iters)

        f = data.create_target_function()
        training_data = [data.create_pair(f, data.create_random_point()) for i in range(N)]
        test_data = [data.create_pair(f, data.create_random_point()) for i in range(test_size)]
        weights, e_in, e_out = compute_avg_error_regression(training_data, test_data)

        e_in_lr_total += e_in
        e_out_lr_total += e_out

        pla_training_data = [data.create_pair(f, data.create_random_point()) for i in range(pla_N)]
        iters_total += pla(weights, pla_training_data)


    print("E_in LR", e_in_lr_total/iters)
    print("E_out LR", e_out_lr_total/iters)
    print("Iters for PLA convergence", iters_total/float(iters))


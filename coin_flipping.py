import random;
import numpy as np;
import math;

total_iters = 100000
c_first_freq_sum = 0.0
c_rand_freq_sum = 0.0
c_min_freq_sum = 0.0
frequencies = np.array([0, 0, 0])

random_integers = np.random.random_integers(0, 999, total_iters)

for i in range(total_iters):
    array = np.random.random_integers(0,1,(1000, 10)).sum(1)
    c_first_freq_sum += array[0]
    c_rand_freq_sum += array[random_integers[i]]
    c_min_freq_sum += array.min()

print(""""frequencies
        first: {0}
        random: {1}
        min: {2}""".format(c_first_freq_sum/(total_iters * 10), c_rand_freq_sum/(total_iters * 10), c_min_freq_sum/(total_iters * 10)))

# single-bin hoeffding
def hoeffding(epsilon, N=10):
    return 2 * math.e ** (-2 * (epsilon ** 2) * N)

def get_bound(delta, mu=0.5):
    return [mu - delta, mu + delta]

print("""hoeffding bounds with different epsilon values:
        0.1: {0}
        0.5: {1}
        0.9: {2}""".format(hoeffding(0.1), hoeffding(0.5), hoeffding(0.9)))


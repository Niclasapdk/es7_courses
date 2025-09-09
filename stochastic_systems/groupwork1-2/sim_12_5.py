import random
N = 10000

inside_ctr = 0

for i in range(N):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if (x*x+y*y) <= 1:
        inside_ctr += 1

prob = inside_ctr / N
pi_est = 4 * prob
print("N: ", N)
print("Probability: ", prob)
print("Estimated pi: ", pi_est)

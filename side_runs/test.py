import numpy as np
f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
exp = np.exp(f)
print("exp: " + str(exp))
np_sum = np.sum(exp)
print("np sum:" ,np_sum)
p = exp / np_sum  # Bad: Numeric problem, potential blowup
print("result:" ,p)
# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
np_exp = np.exp(f)
print("exp: " + str(np_exp))
np_sum = np.sum(np_exp)
print("np sum:" + str(np_sum))
p = np_exp / np_sum  # safe to do, gives the correct answer
print("result:" ,p)

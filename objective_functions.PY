import numpy as np





def obj_intensity(perturbation):
actual_value = np.array([1, 2, 3])
predicted_value = np.array([1.1, 2.1, 5 ])

# take square of differences and sum them
l2_norm = np.sum(np.power((actual_value-predicted_value),2))

# take the square root of the sum of squares to obtain the L2 norm
l2_norm = np.sqrt(l2)
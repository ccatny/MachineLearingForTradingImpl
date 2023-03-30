import numpy as np
np.random.seed(2)
x = np.random.uniform(0, 4)
np.random.seed(2)
y = np.random.uniform(0, 4)
z = np.random.uniform(0, 4)
print(x==y, y==z)
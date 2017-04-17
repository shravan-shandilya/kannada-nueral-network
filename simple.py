import numpy as np
from nueral_network import Nueral_Network

X = np.array(([3,5,5], [5,12,4], [10,2,23]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)
quantisation = np.amax(X,axis=0)
X = X/quantisation
y = y/100

nn = Nueral_Network(3,4,1)
nn.fit(X,y)
print nn.learning
test = np.array(([3,5,5], [5,12,4], [2,1,5], [4,2,7],[2,5,1]),dtype=float)
print nn.predict(test/quantisation)

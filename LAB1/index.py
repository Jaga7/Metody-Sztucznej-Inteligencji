# The standard way to import NumPy:
import numpy as np
import pandas 
import sklearn


tab=np.ones((3,5))
print(tab)

print(np.sum(tab,axis=0))

print(np.sum(tab,axis=1))

x = np.arange(20)

print(x)

print(x[10:13])
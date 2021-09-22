import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
data = pd.read_csv("iris.csv")
   
plt.hist(data["sepallength"])
plt.show()
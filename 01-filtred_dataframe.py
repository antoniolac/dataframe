#nominal import

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pickle import TRUE

#load dataframe
dataframe = pd.read_csv("pearson_dataset.csv")
print("\ndataframe in inch\n", dataframe)

#inch to cm
dataframe["fheight"] = dataframe["fheight"] *2.54
dataframe["sheight"] = dataframe["sheight"] *2.54
print("\n\ndataframe in cm\n", dataframe)

#selection only sheight>170
selection1 = dataframe[dataframe["sheight"]> 170]
print("\n\height sheight > 170cm\n", selection1)

#sort sheight
dataframe.sort_values(by="sheight", ascending=False, inplace=True)
print("\n\nsheight in descending order", dataframe)

#select fheight>180 e sheight<170
selection2 = dataframe[(dataframe["fheight"] > 180) & (dataframe["sheight"] < 170)]
print("\n\nselection 2\n", selection2)


#null check
print("\n\nnull in the dataset\n", dataframe.isna().sum())


#ScatterPlot
plt.figure(figsize=(8, 6))
plt.scatter(dataframe["fheight"], dataframe["sheight"], alpha=0.5)

plt.xlabel("Father Height (cm)")
plt.ylabel("Son Height (cm)")
plt.title("Relation beetween father height and son height")

plt.show()


#BoxPlot
data = [dataframe["fheight"].dropna(), dataframe["sheight"].dropna()]

plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=["Father Height (fheight)", "Son Height (sheight)"])
plt.title("Distribution")
plt.ylabel("Height (cm)")

plt.show()


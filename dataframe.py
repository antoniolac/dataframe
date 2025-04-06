# import nominali

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pickle import TRUE

# Uso import nominale di pandas per leggere un dataset
dataframe = pd.read_csv("pearson_dataset.csv")
print("\ndataframe in inch\n", dataframe)

#converire da inch a cm
dataframe["fheight"] = dataframe["fheight"] *2.54
dataframe["sheight"] = dataframe["sheight"] *2.54
print("\n\ndataframe in cm\n", dataframe)

#seleziona solo sheight maggiori di 170
selezione1 = dataframe[dataframe["sheight"]> 170]
print("\n\naltezze sheight maggiori di 170cm\n", selezione1)

#ordina sheight in ordine decrescente
dataframe.sort_values(by="sheight", ascending=False, inplace=True)
print("\n\nsheight in ordine decrescente", dataframe)

#selezione fheight>180 e sheight<170
selezione2 = dataframe[(dataframe["fheight"] > 180) & (dataframe["sheight"] < 170)]
print("\n\nselezione 2\n", selezione2)


#controllo numero di null nel dataset
print("\n\nnull nel dataset\n", dataframe.isna().sum())


#scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(dataframe["fheight"], dataframe["sheight"], alpha=0.5, color='teal', edgecolors='k')

plt.xlabel("Altezza del padre (cm)")
plt.ylabel("Altezza del figlio (cm)")
plt.title("Relazione tra altezza del padre e altezza del figlio")

plt.show()


#box plot
dati = [dataframe["fheight"].dropna(), dataframe["sheight"].dropna()]

plt.figure(figsize=(8, 6))
plt.boxplot(dati, labels=["Altezza padre (fheight)", "Altezza figlio (sheight)"], patch_artist=True,
            boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
plt.title("Distribuzione delle altezze di padre e figlio")
plt.ylabel("Altezza (cm)")

plt.show()


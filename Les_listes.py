import os
import pandas

athlete_reroot2 = pandas.read_excel("/Users/utilisateur/Dropbox/École/Hiver 2022/Programmation Python/git/Cours_py_2022/athlete_reroot2.xlsx")
print(athlete_reroot2)

lst1 = athlete_reroot2["age"].tolist()
lst2 = athlete_reroot2["equipe"].tolist()

print(lst1)
print(lst2)

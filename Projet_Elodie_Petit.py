#!/usr/bin/env python
#-*- coding: utf-8 -*-

#  Projet python Élodie Petit (20134205) 

# Importation des librairies 

import pandas
import numpy as np
import seaborn as sns; sns.set()
import openpyxl

import sklearn
from sklearn import impute
from sklearn import model_selection 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.cluster import KMeans

import statsmodels
from statsmodels.formula.api import ols

import scipy  
from scipy import stats


import matplotlib.pyplot as plt

from sqlite3 import connect


# Importation de la base de données 

pandas.set_option('display.max_columns', None)
athlete_reroot = pandas.read_excel(r"athlete_reroot.xlsx")
print(athlete_reroot)


# Utilisation d'une classe : READFILE 


df = pandas.DataFrame(athlete_reroot)
df.to_excel("athlete_re.xlsx", 
    engine = "openpyxl")
athlete_re = pandas.read_excel(r"athlete_re.xlsx")


class READFILE:
    
    def __init__(self, file):
        self.file = file
        if file.endswith("csv"):
            self.read_csv()
        elif file.endswith("xlsx"):
            self.read_excel()

    def read_csv(self):
        return pandas.read_csv(self.file)

    def read_excel(self):
        return pandas.read_excel(self.file)
    
    def verif_presence(self):
        return os.path.exists(self.file)

athlete_reroot1 = READFILE("athlete_re.xlsx")
athlete_reroot1.read_excel()


# Algorithme d'automatisation 

liste = athlete_reroot1.read_excel()["age"].tolist()

 """La fonction recherche_lineaire cherche une valeur dans une liste   
    Args:
        x = la valeur cherchée(str)        
        Table = liste avec des nombres (int)
    Return:                                      
        True si la valeur est dans la liste 
    """
def recherche_lineaire(x, Table):
        
    if x in Table:

            reponse = True

    print(reponse) 

recherche_lineaire("23yo", liste)


"""Avec l'algorithme recherche_lineaire il est possible d'affirmer qu'il y a un item nommé
23yo dans la base de données. Il va falloir le retirer avant de faire les analyses."""


"""Afin de voir rapidement où se trouve le 23yo j'utilise la fonction anonyme zip pour 
regarder dans deux colonnes en même temps. Les listes sont dans un autre document
alors j'utilise la pipeline pour aller les chercher."""


# Pipeline

from fonction_anonyme import Les_listes

age = Les_listes.lst1
equipe = Les_listes.lst2


# Fonction anonyme 

lst = zip(age, equipe)
print(list(lst))


# Pré-traitement de la base de données, modification du 23yo  

#Pour vérifier si dans les colonnes il y a des valeurs en string

for colonne in athlete_reroot:
    """Cette boucle permet de regarder chaque colonne de la base de données"""
    for i in athlete_reroot[colonne]:
        """Cette boucle permet de regarder chez item dans une colonne
            si l'item est un string on analyse l'item."""
        if isinstance(i,str):
            if any(k.isdigit() for k in i):
                a = i
                for k in i:
                    """Cette boucle regarde chaque caractère de l'item
                        si le caractère n'est pas un chiffre, on le retire (ici on le remplace par le vide)"""
                    if not k.isdigit():
                        a = a.replace(k, "")   
                athlete_reroot.replace(i, int(a), inplace = True) 
                """Pour changer la nouvelle valeur dans la base de données"""

print(athlete_reroot)


# Remplacer les valeurs manquantes -999 par NAN

headers = athlete_reroot.columns

#Remplacer tous les -999 par NAN
for i in athlete_reroot.columns:
    athlete_reroot[i] = athlete_reroot[i].replace(-999,np.nan)

#Changer tous les NAN par la moyenne de la colonne 
imp = impute.SimpleImputer(missing_values = np.nan , 
                           strategy = "mean")

athlete_reroot2 = pandas.DataFrame(imp.fit_transform(athlete_reroot), 
                                   columns = headers)

print(athlete_reroot2)

#Pour sauvegarder la nouvelle base de données athlete_reroot2 en excel dans l'ordinateur :
path = r"athlete_reroot2.xlsx"
writer = pandas.ExcelWriter(path, engine = 'xlsxwriter')
athlete_reroot2.to_excel(writer)
writer.save()


# Vérification de la base de données - Fonction gestion d'erreurs

'''Pour vérifier si la gestion des erreurs a bien été faite, j'utilise
la fonction de gestion d'erreur nommée recherchestring '''

age = athlete_reroot2["age"].tolist()

 """La fonction recherchestring regarde s'il reste des strinfs dans la base de données  
    Args: 
        liste = liste avec des nombres (int)
    Return:                                      
        oui si la valeur n'est pas un string
    """
def recherchestring(liste):
    
    for index in range (len(liste)):
        if isinstance(liste[index], float):
            print(liste[index])
            
            try:
                index != str
                print("oui")
            
            except ValeurErreur as e:
                print("erreur:", e)
    
recherchestring(age)

""" Toutes les valeurs sortent "oui" alors il n'y a plus de string dans la base de données"""

# Régression linéaire (statistique 1)
# Faire la régression linéaire, est-ce que le bien-être au temps 1 prédit la performance au temps 1 ?

model = ols("t1_perfo ~ t1_bienetre", athlete_reroot2).fit()

print(model.summary())

t = model.tvalues[1]
p = model.pvalues[1]
r = model.rsquared

print(f"""Rapport de la régression linéaire: La valeur t du bien-être au temps 1 est de {t:.3f} 
et la valeur p est de {p:.3f}. Ainsi, bien-être au temps 1 a bel est bien un impact statistiquement significatif
de prédiction sur la performance au temps 1. Toutefois, le R carré est de {r:.3f} ce qui indique que la qualité 
de la régression n'est pas excellente""")   


# Visualisation de la régression linéaire 
# Graphique pour régression : pour montrer si bien-être au temps 1 prédit la performance au temps 1

ax = sns.regplot(data = athlete_reroot2, 
                 x = "t1_bienetre", 
                 y = "t1_perfo", 
                 marker = "+")

ax.set(xlim=(2,7), ylim=(1,7), 
       xlabel='Bien-être au temps 1', 
       ylabel='Performance au temps 1',
       title='Performance au temps 1 en fonction du bien-être au temps 1')

plt.show()


# Test-T (statistique 2)
# Faire le test-t apparié: comparer le bien-être des athlètes entre le T1 et le T2 

TestT= stats.ttest_rel(athlete_reroot2["t1_bienetre"],
                       athlete_reroot2["t2_bienetre"])

t = TestT.statistic
p = TestT.pvalue

print(f"""Rapport du Test-T: La différence entre le bien-être au temps 1 et le bien-être 
au temps 2 n'est pas significatif avec une valeur de t de {t:.3f} et une valeur de p de {p:.3f}. 
Ainsi il n'y a pas de différence entre les deux groupes. La figure qui suit va bien démontrer le tout.""") 


# Visualisation du Test-T graphique

athlete_reroot3 = athlete_reroot2[["t1_bienetre","t2_bienetre"]]
x = pandas.melt(athlete_reroot3, 
                id_vars = None, 
                var_name ='Temps', 
                value_name ='Bien-être')

with sns.axes_style(style ='ticks'):
    ax = sns.catplot(data = x, 
                     x = 'Temps', 
                     y = 'Bien-être',
                     hue = None, 
                     kind ='box')
    ax.set(title ='Comparaison du bien-être au temps 1 et au temps 2')

plt.show()


# Apprentissage machine supervisée et validation croisée 

# Définition de X et Y 

X_t1_bienetre = [athlete_reroot2["t1_bienetre"],
                athlete_reroot2["t1_perfo"]]

headers = ["t1_bienetre", "t1_perfo"]
X1 = pandas.concat(X_t1_bienetre,axis = 1, keys = headers)
X = X1.to_numpy()

y1 = athlete_reroot2["uni"]
y = y1.to_numpy()

# Choix de l'algorithme et de ses paramètres
random_state = np.random.RandomState(0)
cv = sklearn.model_selection.StratifiedKFold(n_splits = 6)
classifier = sklearn.svm.SVC(kernel = 'linear',
                             probability = True,
                             random_state = random_state)

# Validation croisée 
prediction_accuracy = []
for i, (train, test) in enumerate(cv.split(X, y)):
    Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X, y)  
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    pipe.fit(X[train], y[train]) 
    y_model = pipe.predict(Xtest)
    prediction_accuracy.append(sklearn.metrics.accuracy_score(ytest, y_model)) 

print(f"""Rapport apprentissage supervisée: Apprentissage machine supervisée avec agorithme SVC a en moyenne 
      une précision de prédiction de {p:.2f} ce qui est relativement bon. Il est possible de relativement bien 
      déterminer l'université de l'athlète selon son bien-être et sa performance au temps 1""")


# Visualisation de l'apprentissage supervisée

for i, (train, test) in enumerate(cv.split(X, y)):
    Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X, y)

    classifier = sklearn.svm.SVC(kernel = 'linear',
                             probability = True,
                             random_state = random_state)
    classifier.fit(X[train], y[train])

disp = plot_confusion_matrix(classifier, Xtest, ytest, cmap = "Blues")

plt.show()


# Apprentissage non-supervisée et sa visualisation 

X = athlete_reroot2.iloc[0:, 0:86].to_numpy()

def kmeans(X):
    
    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 50, cmap = "viridis")
    
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c = "black", s = 200, alpha = 0.5);
    
    resultat = sklearn.metrics.accuracy_score(athlete_reroot2["uni"], y_kmeans)
    print(f"""Rapport apprentissage non-supervisé: L'apprentissage machine non-supervisé avec l'algorithme kmeans donne une 
    précision de prédiction de {resultat:.3f}. Il est possible de relativement bien 
      déterminer à quelle université appartiennent les athlètes. """)

kmeans(athlete_reroot2.iloc[0:, 0:86].to_numpy())
plt.show()


# SQLITE 
conn = connect('new.db', check_same_thread = False)

def __connect_db():
    conn = connect('new.db', check_same_thread = False)
    return conn

def __creer_tableau_(tableau):
    conn = __connect_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS {0} (ID, uni, genre, age)'''
                 .format(tableau,))
    conn.commit()
    conn.close()
    

def __definir_les_donnees_(tableau, donnees):
    conn = __connect_db()
    conn.execute('''INSERT INTO {0} VALUES {1}'''.format(tableau, donnees))
    conn.commit()
    conn.close()
    
__creer_tableau_('ATHLETE')
__definir_les_donnees_('ATHLETE', (63, 1, 1, 24))
print(conn.execute("""SELECT * FROM ATHLETE WHERE age = 24 """).fetchall())









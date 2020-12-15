

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pickle
import codecs



filename = "W:\Documents\data.csv"

# Importation de la base de données

data = pd.read_csv(filename)

# Nettoayge de la base data

#transformer la base en 2 bases categorique et numerique

cat_data=[]

num_data=[]

for i,c in enumerate(data.dtypes):

  if c==object:

    cat_data.append(data.iloc[:,i])

  else:

    num_data.append(data.iloc[:,i])

cat_data=pd.DataFrame(cat_data).transpose()

num_data=pd.DataFrame(num_data).transpose()

#modifier les valeurs de la colonne caesarian de type catégorique au type numérique

Caesarian1_value={'yes':1,'Yes':1,'No':0}

Caesarian1=cat_data['Caesarian']

cat_data.drop('Caesarian',axis=1, inplace=True)

Caesarian1=Caesarian1.map(Caesarian1_value)

#modifier les valeurs de la colonne pressureblood

PressBlood_value={'Low':0,'low':0,'High':2,'Normal':1}

PressBlood=cat_data['Blood of Pressure']

cat_data.drop('Blood of Pressure',axis=1, inplace=True)

PressBlood=PressBlood.map(PressBlood_value)


# Transformation des labels catégoriels en labels numériques


le=LabelEncoder()

for i in cat_data:

  cat_data[i] = le.fit_transform(cat_data[i])

data1=pd.concat([cat_data,num_data,PressBlood,Caesarian1],axis=1)


# Affichage de la table descriptive


Table_descriptive = data1.describe()


print(Table_descriptive)

boxprops = dict(linestyle='-', linewidth=3, color='darkgoldenrod')

plt.figure(0)

boxplot = data1.boxplot(column='Caesarian', boxprops=boxprops)

boxplot = data1.boxplot(column='Delivery No', boxprops=boxprops)

boxplot = data1.boxplot(column='Blood of Pressure', boxprops=boxprops)



# A travers les boites de moustache des différentes variables on peut voir que les variables sont distribuées à peu près dans des intervalles de meme longueur, un point important aussi c'est que les variables Delivey No, Heart Problem et Caesarian sont asymétriques donc leurs données peuvent ne pas être normalement distribuées et concernant les valeurs abberantes notre graphe n'affiche pas des données de cette nature pour toutes les variables de traitement.

#effets des variables sur la variable Caesarian

#effet de la variable Age

grid=sns.FacetGrid(data1,col='Caesarian',size=3.2,aspect=1.6)

grid.map(sns.countplot,'Age')

#l'operation est encouragée pour les femmes autour des trentaines et découragé pour celles autour des vingtaines
#l'age a un effet sur la variable caesarian

#effet de la varible Delivey No

grid=sns.FacetGrid(data1,col='Caesarian',size=3.2,aspect=1.6)

grid.map(sns.countplot,'Delivey No')

#on déduit que le nombre  d'opérations cesariennes est plus élevé chez les femmes ayant moins d'enfants et surtout lorsqu'il s'agit de leur premier enfant
#mais aussi non necessité d'operation pour les femmes ayant moins d'enfants
#conclusion: la variable nbrenfants n'a pas un effet majeur sur l'accord d'operation cesarienne

#effet de la variable Heart Problem

grid=sns.FacetGrid(data1,col='Caesarian',size=3.2,aspect=1.6)

grid.map(sns.countplot,'Heart Problem')

#pas d'effet majeur sur la decision d'operation cesarienne selon la variable HeartProblem
#mais les femmes n'ayant pas de probleme cardiovasculaire effectuent moins d'operations

#effet de la variable Delivery No

grid=sns.FacetGrid(data1,col='Caesarian',size=3.2,aspect=1.6)

grid.map(sns.countplot,'Delivery No')

#plus les femmes auront une grossesse retardée plus elles effectuent une operation
#conclusion: la variable Deliveru No affecte la variable Caesarian

#effet de la variable Blood of Pressure

grid=sns.FacetGrid(data1,col='Caesarian',size=3.2,aspect=1.6)

grid.map(sns.countplot,'Blood of Pressure')

#les femmes ayants une pression arterielle normale effectuent moins d'operations cesariennes
#la pression arterielle n'a pas d'effet sur les femmes ayants effectué une operation

#on peut aussi faire une autre analyse avec les medianes afin de prouver des corrélations

data1.groupby('Caesarian').median()

#donc on peut voir une correlation entre l'age est la decision de l'operation telque les operations sont acceptées plus pour les femmes les plus agées
#Mais aussi pour les grossesses retardées comme évoqué précedement et à lesquelles s'ajoute les problemes cardiovasculaires mais la variable Blood of Pressure reste
#sans effet sur la decison d'operation

# Prédiction par regression



# Prédiction par KNN

from sklearn.neighbors import KNeighborsClassifier


y = data1["Caesarian"]

X = data1.copy()

X.drop(["Caesarian"], axis=1, inplace=True)

train_size = 0.8  # train -> 80 %, Test -> 20 %


# Division des données en base d'apprentissage et en base de test

N_train = int(train_size*len(y))
N_test = len(y) - N_train


X = X.to_numpy()
y = y.to_numpy()

X_train = X[0:N_train,:]
X_test = X[N_train:,:]


y_train = y[0:N_train]
y_test = y[N_train:]

from sklearn.neighbors import NearestNeighbors
NN = NearestNeighbors(n_neighbors=2)


NN.fit(X)

A = NN.kneighbors_graph(X)

# Instance d'un KNN
neigh = KNeighborsClassifier(n_neighbors=3)

# COnstruction du modèle KNN sur la base d'apprentissage
neigh.fit(X_train, y_train)

predictions = neigh.predict(X_test)



from sklearn.metrics import roc_curve





from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot


'''
------------------------------------------------------------------------------
                    Regression logistique
------------------------------------------------------------------------------
'''



# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2)


# fit a model


model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)


# predict probabilities
yhat = model.predict_proba(testX)

# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]


plt.figure(1)

# calculate roc curve for model
fpr, tpr, _ = roc_curve(testy, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='ROC Regression logistique')
# axis labels
pyplot.xlabel('Taux False Positive')
pyplot.ylabel('Taux True Positive')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# La courbe de la régression logistique ne ressemble pas totalement à sinusoide donc cette régression ne peut pas expliquer notre modèle avec l'ensemble de variables d'entrée qu'on dispose mais ca ser intéressant de l'étudier si on ajoute plus de varibles d'entrée.

'''
------------------------------------------------------------------------------
                    KNN classifieur
------------------------------------------------------------------------------
'''


model = KNeighborsClassifier(n_neighbors=3)
model.fit(trainX, trainy)


# predict probabilities
yhat = model.predict_proba(testX)

# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]


plt.figure(1)

# calculate roc curve for model
fpr, tpr, _ = roc_curve(testy, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='ROC KNN classifieur')
# axis labels
pyplot.xlabel('Taux False Positive')
pyplot.ylabel('Taux True Positive')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

#La courbe ressemble plus à une courbe linéaire ce qui est plus pertinente pour expliquer la classification de nos données


'''
------------------------------------------------------------------------------
                    Arbre de décision
------------------------------------------------------------------------------
'''



from sklearn.tree import DecisionTreeClassifier


model = DecisionTreeClassifier(random_state=0)



model.fit(trainX, trainy)


# predict probabilities
yhat = model.predict_proba(testX)

# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]


plt.figure(1)

# calculate roc curve for model
fpr, tpr, _ = roc_curve(testy, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='ROC Arbre de décision')
# axis labels
pyplot.xlabel('Taux False Positive')
pyplot.ylabel('Taux True Positive')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# la courbe de l'arbre de décision contient une discontinueté qui met en défaut la capacité de l'algorithme à prédire le modèle due à un manque d'information sur le dernier résultat du modèle qui reste incompréhensible par cet algorithme car la courbe devient verticale à la valeur 1 d'abcisse




'''
------------------------------------------------------------------------------
                    Foret aléatoire
------------------------------------------------------------------------------
'''

from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(max_depth=2, random_state=0)


model.fit(trainX, trainy)


# predict probabilities
yhat = model.predict_proba(testX)

# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]


plt.figure(1)

# calculate roc curve for model
fpr, tpr, _ = roc_curve(testy, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='ROC Forêt aléatoire')
# axis labels
pyplot.xlabel('Taux False Positive')
pyplot.ylabel('Taux True Positive')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

#Par manque de données pour ce modèle il affiche plus de ligne de verticale dans la courbe donc l'algorithme de doret aleatoire n'est pas pertinent pour notre prédiction

'''
------------------------------------------------------------------------------
                  Perception multi-couche
------------------------------------------------------------------------------
'''




from sklearn.neural_network import MLPClassifier


model = MLPClassifier(random_state=1, max_iter=300)


model.fit(trainX, trainy)


# predict probabilities
yhat = model.predict_proba(testX)

# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]


plt.figure(1)

# calculate roc curve for model
fpr, tpr, _ = roc_curve(testy, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='ROC Perception multi-couche')
# axis labels
pyplot.xlabel('Taux False Positive')
pyplot.ylabel('Taux True Positive')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


#Après analyse de toutes les courbes des différents algorithmes de machine learning il semble que l'algorithme de KNN est le plus pertinent à  predire notre modèle avec une régression presque linéaire et qui sera amélioré si on restreint notre test sur l'ensemble des variables qui affectent le plus notre résultat comme indiqué au dessus. Donc on va choisir l'algorithme KNN pour le déploiment de notre application finale.

#appliquer le modele du KNN sur notre base de données
model= KNeighborsClassifier(n_neighbors=3)
model.fit(data1,Caesarian1)

#enregistrer notre modele pour le deploiment dans l'application
pickle.dump(model,open('model.pkl','wb'))


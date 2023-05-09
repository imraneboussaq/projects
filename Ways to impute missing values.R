# Chargement des packages nécessaires

library(tidyverse) #Ensemble de packages pour la manipulation et la visualisation de données, y compris dplyr, ggplot2, tidyr, etc.
library(mice) #Package pour l'imputation multiple des données manquantes, "Multivariate Imputation by Chained Equations".
library(randomForest) #Package pour la création de modèles de forêts aléatoires.
library(rpart) #Package pour la création de modèles d'arbres de décision récursifs.
library(class) #Package pour la classification basée sur des modèles de k plus proches voisins (k-NN) "K-Nearest Neighbors" .
library(impute) #Package pour l'imputation des données manquantes.
library(corrplot) #Package pour la création de graphiques de corrélation.
library(dplyr) #Package pour la manipulation de données, y compris la sélection, le filtrage, la transformation et la synthèse.
library(magrittr) #Package pour la manipulation de données en utilisant des pipes (%>%).
library(caret) #Package pour l'apprentissage automatique et la modélisation prédictive.
library(imputeMissings) #Package pour l'imputation de données manquantes.

library(VIM) # (Visualization and Imputation of Missing Values), propose plusieurs approches pour imputer les données manquantes en utilisant les algorithmes de régression, k-NN
library(missForest)
library(MASS)
library(DMwR2)



# Chargement de la base de données avec des valeurs manquantes
housing <- read.csv("HousingData.csv") #J'ai modifié la trajectoire pour que le fichier soit lu à partir d'un autre emplacement


#CRIM - per capita crime rate by town
#ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
#INDUS - proportion of non-retail business acres per town.
#CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
#NOX - nitric oxides concentration (parts per 10 million)
#RM - average number of rooms per dwelling
#AGE - proportion of owner-occupied units built prior to 1940
#DIS - weighted distances to five Boston employment centres
#RAD - index of accessibility to radial highways
#TAX - full-value property-tax rate per "$10,000"
#PTRATIO - pupil-teacher ratio by town
#B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#LSTAT - lower status of the population
#MEDV - Median value of owner-occupied homes in $1000's



# Visualisation des données pour avoir une idée de leur distribution
# Utilisation des fonctions `summary()`, `head()` et `str()` pour afficher des informations sur les données
head(housing)
summary(housing)
str(housing)


# Conversion des colonnes pertinentes en format numérique pour une analyse plus facile
housing$CHAS=as.numeric(housing$CHAS)
housing$RAD=as.numeric(housing$RAD)
housing$TAX=as.numeric(housing$TAX)


# visualiser le nombre de valeurs manquantes dans chaque colonne de la base de données "housing"
sapply(housing, function(x) sum(is.na(x)))


# Chargement de la base de données Bostonhousing, sans NAs (complete)
data(Boston)

#Affichons une matrice de corrélation du dataset 
corrMatrix <- cor(Boston) #calculer les corrélations entre les différentes variables
corrplot(corrMatrix, type = 'upper', order = 'hclust', tl.col = 'black', tl.srt = 45) #afficher la matrice de corrélation



#################################

#################################









################################
# Suppression des observations contenant des valeurs manquantes
################################

data_no_na <- na.omit(housing)
sum(is.na(data_no_na))
 


#diviser les données en deux variables: X et y.
X <- data_no_na[, !names(data_no_na) %in% c('CHAS', 'MEDV')] #X contient toutes les colonnes de la base de données housing sauf CHAS et MEDV
#la variable CHAS (qui indique si la propriété est située le long de la Charles River) n'est pas pertinente pour la tâche en cours.
#elle est fortement corrélée à d'autres variables qui ont été incluses dans l'analyse
y <- data_no_na$MEDV #y contient uniquement la colonne MEDV
#Cela permet de séparer les variables explicatives des variables à prédire.



#diviser le jeu de données en un ensemble d'entraînement et un ensemble de test.
set.seed(124) # pour initialiser une séquence de nombres aléatoires dans R, Cela est particulièrement utile lors de la division 
#d'un ensemble de données en un ensemble d'entraînement et un ensemble de test, car cela garantit que les mêmes observations 
#sont sélectionnées pour chaque ensemble à chaque exécution du code.
train_index <- sample(nrow(data_no_na), round(nrow(data_no_na) * 0.7)) #diviser le jeu de données en un ensemble d'entraînement et un ensemble de test.
#avec une proportion de 70% pour l'ensemble d'entraînement et 30% pour l'ensemble de test.
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]



################################
#La méthode d'imputation de moyenne
################################

# Chargement de la base de données avec des valeurs manquantes
housing <- read.csv("HousingData.csv") 

# Conversion des colonnes pertinentes en format numérique pour une analyse plus facile
housing$CHAS=as.numeric(housing$CHAS)
housing$RAD=as.numeric(housing$RAD)
housing$TAX=as.numeric(housing$TAX)

#diviser les données en deux variables: X et y.
X <- housing[, !names(housing) %in% c('CHAS', 'MEDV')] 
y <- housing$MEDV 


#diviser le jeu de données en un ensemble d'entraînement et un ensemble de test.
set.seed(124) 
train_index <- sample(nrow(housing), round(nrow(housing) * 0.7)) 
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]


X_train_filled_mean <- X_train %>% mutate(across(where(is.numeric), ~if_else(is.na(.), mean(., na.rm = TRUE), .))) #imputer les valeurs manquantes dans les variables numériques des données d'apprentissage.
mean_imputed_model <- lm(y_train ~ ., data = X_train_filled_mean) #un modèle de régression linéaire est ajusté aux données d'apprentissage imputées avec la moyenne 
X_test_filled_mean <- X_test %>% mutate(across(where(is.numeric), ~if_else(is.na(.), mean(., na.rm = TRUE), .))) #imputer les valeurs manquantes dans les variables numériques des données de test.
y_pred_mean <- predict(mean_imputed_model, X_test_filled_mean) #utiliser le modèle ajusté pour prédire les valeurs de la variable cible y_test.


# calculer la RMSE (root mean square error) pour un modèle de régression linéaire multiple.
RMSE_mean_model <- sqrt(mean((y_pred_mean - y_test) ^ 2))
cat('Le RMSE du score après imputation des valeurs manquantes avec la moyenne:', RMSE_mean_model, '\n')




################################
## Remplacer les valeurs manquantes dans toutes les variables numériques par leur médiane
################################

# Chargement de la base de données avec des valeurs manquantes
housing <- read.csv("HousingData.csv") 

# Conversion des colonnes pertinentes en format numérique pour une analyse plus facile
housing$CHAS=as.numeric(housing$CHAS)
housing$RAD=as.numeric(housing$RAD)
housing$TAX=as.numeric(housing$TAX)

#diviser les données en deux variables: X et y.
X <- housing[, !names(housing) %in% c('CHAS', 'MEDV')] 
y <- housing$MEDV 

#diviser le jeu de données en un ensemble d'entraînement et un ensemble de test.
set.seed(124) 
train_index <- sample(nrow(housing), round(nrow(housing) * 0.7)) 
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

X_train_filled_median <- X_train %>% mutate(across(where(is.numeric), ~if_else(is.na(.), median(., na.rm = TRUE), .))) #imputer les valeurs manquantes dans les variables numériques des données d'apprentissage.
median_imputed_model <- lm(y_train ~ ., data = X_train_filled_median) #un modèle de régression linéaire est ajusté aux données d'apprentissage imputées avec la mediane 
X_test_filled_median <- X_test %>% mutate(across(where(is.numeric), ~if_else(is.na(.), median(., na.rm = TRUE), .))) #imputer les valeurs manquantes dans les variables numériques des données de test.
y_pred_median <- predict(median_imputed_model, X_test_filled_median) #utiliser le modèle ajusté pour prédire les valeurs de la variable cible y_test.



# calculer la RMSE (root mean square error) pour un modèle de régression linéaire multiple.
RMSE_median_model <- sqrt(mean((y_pred_median - y_test) ^ 2))
cat('Le RMSE du score après imputation des valeurs manquantes avec la médiane:', RMSE_median_model, '\n')



###################################
# Imputer les valeurs manquantes avec la méthode de régression multiple
###################################

# Charger les données
df <- read.csv("HousingData.csv")

# Conversion des colonnes pertinentes en format numérique pour une analyse plus facile
df$CHAS=as.numeric(df$CHAS)
df$RAD=as.numeric(df$RAD)
df$TAX=as.numeric(df$TAX)


df[] <- lapply(df, as.numeric) # Convertir toutes les colonnes en type numérique

imp <- mice(df, method="norm.predict", m=5, maxit=50)#la fonction mice est utilisée pour imputer les valeurs manquantes en utilisant la méthode norm.predict, qui est une méthode de régression multiple. 
                                                     #Les options m=5 et maxit=50 spécifient le nombre d'itérations et le nombre d'ensembles de données à générer pour l'imputation.
                                                     #La fonction complete est utilisée pour combiner les résultats imputés en un seul ensemble de données.
# Combiner les résultats
data_regression <- complete(imp)



# Diviser les données en ensembles d'entraînement et de test
train_index <- sample(nrow(data_regression), round(nrow(data_regression) * 0.7))
X_train <- data_regression[train_index, -ncol(data_regression)]
X_test <- data_regression[-train_index, -ncol(data_regression)]
y_train <- data_regression[train_index, ncol(data_regression)]
y_test <- data_regression[-train_index, ncol(data_regression)]

# Entraîner le modèle
mice_imputed_model <- lm(y_train ~ ., data = X_train)

# Faire des prédictions sur l'ensemble de test
y_pred_mice <- predict(mice_imputed_model, X_test)

# Calculer le score RMSE
RMSE_mice_model <- sqrt(mean((y_pred_mice - y_test) ^ 2))
cat('Le RMSE du score après imputation des valeurs manquantes avec la regression multiple:', RMSE_mice_model, '\n')






##############################
# Impute with KNNImputer
##############################


# Charger les données
data <- read.csv("HousingData.csv")

# Conversion des colonnes pertinentes en format numérique pour une analyse plus facile
data$CHAS=as.numeric(data$CHAS)
data$RAD=as.numeric(data$RAD)
data$TAX=as.numeric(data$TAX)

# Imputer les valeurs manquantes avec KNN
data_KNN <- knnImputation(data)

# Diviser les données en ensembles d'entraînement et de test
train_index <- sample(nrow(data_KNN), round(nrow(data_KNN) * 0.7))
X_train <- data_KNN[train_index, -ncol(data_KNN)]
X_test <- data_KNN[-train_index, -ncol(data_KNN)]
y_train <- data_KNN[train_index, ncol(data_KNN)]
y_test <- data_KNN[-train_index, ncol(data_KNN)]

# Entraîner le modèle
knn_imputed_model <- lm(y_train ~ ., data = X_train)

# Faire des prédictions sur l'ensemble de test
y_pred_knn <- predict(knn_imputed_model, X_test)

# Calculer le score RMSE
RMSE_knn_model <- sqrt(mean((y_pred_knn - y_test) ^ 2))
cat('Le RMSE du score après imputation des valeurs manquantes avec KNN:', RMSE_knn_model, '\n')




#################################
# Imputation des valeurs manquantes arbre de decision
#################################

df <- read.csv("HousingData.csv", header = TRUE)

# Conversion des colonnes pertinentes en format numérique pour une analyse plus facile
df$CHAS=as.numeric(df$CHAS)
df$RAD=as.numeric(df$RAD)
df$TAX=as.numeric(df$TAX)

for (i in 1:ncol(df)) {
  if (sum(is.na(df[, i]))) {
    formula <- as.formula(paste0(names(df)[i], " ~ ."))
    tree_model <- rpart(formula = formula, data = na.omit(df), method = "anova")
    imputed_values <- predict(tree_model, newdata = df)
    df[is.na(df[, i]), i] <- imputed_values[is.na(df[, i])]
  }
}
# La boucle for permet de parcourir toutes les colonnes du dataset et de vérifier si elles contiennent des valeurs manquantes. 
#Si c'est le cas, une formule est créée pour construire un arbre de décision en utilisant les autres variables du dataset comme 
#prédicteurs pour la variable manquante. Ensuite, les valeurs manquantes sont remplacées par les valeurs prédites par l'arbre de décision. 
#La fonction rpart du package rpart est utilisée pour construire l'arbre de décision. La méthode utilisée est "anova".
sapply(df, function(x) sum(is.na(x)))


# Diviser les données en ensembles d'apprentissage et de test
set.seed(123)
n_obs <- nrow(df)
train_index <- sample(n_obs, n_obs * 0.7)
X_train <- df[train_index, -14] # toutes les colonnes sauf la dernière qui est la variable cible
y_train <- df[train_index, 14]  # la dernière colonne qui est la variable cible
X_test <- df[-train_index, -14]
y_test <- df[-train_index, 14]

# Entraîner le modèle
decision_tree_imputed_model <- lm(y_train ~ ., data = X_train)

# Faire des prédictions sur l'ensemble de test
y_pred_decision_tree <- predict(decision_tree_imputed_model, X_test)

# Calculer le score RMSE
RMSE_decision_tree_model <- sqrt(mean((y_pred_decision_tree - y_test) ^ 2))

cat('Le RMSE du score après imputation des valeurs manquantes avec arbre de decision:', RMSE_decision_tree_model, '\n')



###############################
#forêt aléatoire
###############################
#Cela implique la création de plusieurs arbres de décision pour prédire les valeurs manquantes 
#en utilisant les autres variables comme prédicteurs. Chaque arbre donne une prédiction différente, 
#et la moyenne de ces prédictions est utilisée comme valeur imputée pour la variable manquante. 
#La méthode de forêt aléatoire peut être plus efficace que les arbres de décision seuls, 
#car elle peut gérer des relations non linéaires et des interactions entre les variables.


# Chargement de la base de données avec des valeurs manquantes
df <- read.csv("HousingData.csv")

# Conversion des colonnes pertinentes en format numérique pour une analyse plus facile
df$CHAS <- as.numeric(df$CHAS)
df$RAD <- as.numeric(df$RAD)
df$TAX <- as.numeric(df$TAX)

# Remplacer les valeurs manquantes avec `missForest`
imputed_data <- missForest(df)

# Convertir la sortie de `missForest` en dataframe
imputed_data <- as.data.frame(imputed_data$ximp)

# Vérifier s'il reste des valeurs manquantes
sapply(imputed_data, function(x) sum(is.na(x)))

# Diviser les données en ensembles d'apprentissage et de test
set.seed(123)
n_obs <- nrow(imputed_data)
train_index <- sample(n_obs, n_obs * 0.8)
X_train <- imputed_data[train_index, -14] # toutes les colonnes sauf la dernière qui est la variable cible
y_train <- imputed_data[train_index, 14]  # la dernière colonne qui est la variable cible
X_test <- imputed_data[-train_index, -14]
y_test <- imputed_data[-train_index, 14]

# Créer et ajuster le modèle de régression multiple
imputed_model <- lm(y_train ~ ., data = X_train)
y_pred_imputed <- predict(imputed_model, X_test)

# Calculer le score RMSE
RMSE_imputed_model <- sqrt(mean((y_pred_imputed - y_test) ^ 2))
cat('Le RMSE du score après imputation des valeurs manquantes avec missForest :', RMSE_imputed_model, '\n')



################################
#deux base de données
################################

# Chargement de la base de données Bostonhousing, sans NAs (complete)
data(Boston)

head(Boston)
summary(Boston)
str(Boston)

Boston$chas=as.numeric(Boston$chas)
Boston$rad=as.numeric(Boston$rad)


# Chargement des données avec NAs () et des données complètes (data2)
data1 <- read.csv("HousingData.csv", header = TRUE)

# Conversion des colonnes pertinentes en format numérique pour une analyse plus facile
data1$CHAS=as.numeric(data1$CHAS)
data1$RAD=as.numeric(data1$RAD)
data1$TAX=as.numeric(data1$TAX)



# Renommer toutes les colonnes de "Housing"
data1 <- rename(housing, crim = CRIM, zn = ZN, indus = INDUS, chas =CHAS, nox = NOX, rm = RM, age = AGE, dis = DIS, rad = RAD , tax = TAX, ptratio = PTRATIO, b = B, lstat=LSTAT, medv =MEDV)



sum(is.na(data1))
data2 <- Boston
sum(is.na(data2))

# Remplacement des NAs dans data1 par les valeurs correspondantes dans data2
for (i in 1:ncol(data1)) {
  data1[is.na(data1[, i]), i] <- data2[is.na(data1[, i]), i]
}

sum(is.na(data1))


# Diviser les données en ensembles d'entraînement et de test
train_index <- sample(nrow(data1), round(nrow(data1) * 0.7))
X_train <- data1[train_index, -ncol(data1)]
X_test <- data1[-train_index, -ncol(data1)]
y_train <- data1[train_index, ncol(data1)]
y_test <- data1[-train_index, ncol(data1)]

# Entraîner le modèle
B_imputed_model <- lm(y_train ~ ., data = X_train)

# Faire des prédictions sur l'ensemble de test
y_pred_B <- predict(B_imputed_model, X_test)

# Calculer le score RMSE
RMSE_B_model <- sqrt(mean((y_pred_B - y_test) ^ 2))
cat('Le RMSE du score après imputation des valeurs manquantes à l'aide d'une base complete:', RMSE_B_model, '\n')

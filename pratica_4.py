# Importações
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregar e dividir os dados
wine = load_wine()
X = wine.data
y = wine.target
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
}

# cv=5 fatias na validação cruzada
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')

# Treinamento
grid_search.fit(X_treino, y_treino)


print(f"Os melhores parâmetros encontrados foram: {grid_search.best_params_}")
print(f"A melhor acurácia média (validação cruzada) foi de: {grid_search.best_score_ * 100:.2f}%")

y_pred = grid_search.predict(X_teste)
acuracia_final = accuracy_score(y_teste, y_pred)
print(f"Acurácia no conjunto de teste final: {acuracia_final * 100:.2f}%")

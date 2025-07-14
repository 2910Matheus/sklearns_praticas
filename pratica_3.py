# from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
# Bônus: Visualizando a árvore de decisão
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.3, random_state=42)

arvore = DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

y_pred = arvore.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"A acurácia da Árvore de Decisão foi de: {accuracy * 100:.2f}%")

plt.figure(figsize=(20,10))
plot_tree(arvore, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names, 
          filled=True)
plt.show()

#Importando as bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Passo 2: Carregar os dados
iris = load_iris()
X = iris.data  # As 'features' (medidas das flores)
y = iris.target # O 'target' (a espécie de cada flor, representada por 0, 1 ou 2)

# X seria as perguntas e y o gabarito

# Passo 3: Dividir os dados em Treino e Teste
# Usarei 80% dos dados para treinar o modelo e 20% para testá-lo.
# O 'random_state=42' garante que a divisão seja sempre a mesma, para que possamos reproduzir os resultados.
# test_size = 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Criar e Treinar o modelo
# Usarei os 3 vizinhos mais próximos (k=3)
# Prioriza-se números impar para desempates
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) 

# Passo 5: Fazer previsões com os dados de teste
y_pred = knn.predict(X_test)

# Passo 6: Avaliar o modelo
# Comparamos as previsões (y_pred) com as respostas corretas (y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"O modelo acertou {accuracy * 100:.2f}% das espécies no conjunto de teste!")

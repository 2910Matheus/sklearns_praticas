# ==============================================================================
# IMPORTAÇÕES E PREPARAÇÃO INICIAL
# ==============================================================================
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X = wine.data
y = wine.target
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================================================================
# CENÁRIO 1: SEM ESCALONAMENTO
# ==============================================================================
print("--- CENÁRIO 1: SEM PRÉ-PROCESSAMENTO ---")
knn_sem_escala = KNeighborsClassifier(n_neighbors=5)
knn_sem_escala.fit(X_treino, y_treino)
y_pred_sem_escala = knn_sem_escala.predict(X_teste)
acuracia_sem_escala = accuracy_score(y_teste, y_pred_sem_escala)
print(f"Acurácia do KNN sem escalonamento: {acuracia_sem_escala * 100:.2f}%")
print("-" * 50)


# ==============================================================================
# CENÁRIO 2: COM ESCALONAMENTO (StandardScaler)
# ==============================================================================
print("--- CENÁRIO 2: COM PRÉ-PROCESSAMENTO ---")
# 1. Cria e ajusta o scaler APENAS nos dados de treino
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)

# 2. Apenas transforma os dados de teste com o scaler já ajustado
X_teste_scaled = scaler.transform(X_teste)

# 3. Treina o KNN com os dados escalonados
knn_com_escala = KNeighborsClassifier(n_neighbors=5)
knn_com_escala.fit(X_treino_scaled, y_treino)

# 4. Faz previsões e avalia
y_pred_com_escala = knn_com_escala.predict(X_teste_scaled)
acuracia_com_escala = accuracy_score(y_teste, y_pred_com_escala)
print(f"Acurácia do KNN COM escalonamento: {acuracia_com_escala * 100:.2f}%")
print("-" * 50)

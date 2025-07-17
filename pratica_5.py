# ==============================================================================
# IMPORTAR AS FERRAMENTAS
# ==============================================================================
from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, r2_score


# ==============================================================================
# CARREGAR E DIVIDIR OS DADOS
# ==============================================================================
diabetes = load_diabetes()
X = diabetes.data 
y = diabetes.target 
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)


# ==============================================================================
# CRIAR E TREINAR O MODELO
# ==============================================================================
regressor = LinearRegression()


regressor.fit(X_treino, y_treino)


# ==============================================================================
# FAZER PREVISÕES E AVALIAR
# ==============================================================================
y_pred = regressor.predict(X_teste)

mae = mean_absolute_error(y_teste, y_pred)
r2 = r2_score(y_teste, y_pred)

print("Resultados da Avaliação do Modelo de Regressão Linear:")
print("-" * 50)
print(f"Erro Médio Absoluto (MAE): {mae:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

print("\n--- Analisando o Modelo por Dentro ---")

print(f"Intercepto (b): {regressor.intercept_:.2f}")

for i, coef in enumerate(regressor.coef_):
    print(f"Coeficiente para a Característica {i+1}: {coef:.2f}")

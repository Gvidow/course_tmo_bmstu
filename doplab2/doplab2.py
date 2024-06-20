import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
housing = fetch_california_housing()
X, y = housing.data, housing.target
columns = housing.feature_names

# Создание боковой панели для выбора гиперпараметров
st.sidebar.header("Гиперпараметры")
n_estimators = st.sidebar.slider("Количество деревьев", 10, 200, 100)
max_depth = st.sidebar.slider("Максимальная глубина дерева", 2, 20, 10)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик качества
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Визуализация результатов
st.title("Random Forest Regression: Moscow housing")
st.write("""
**Задача:** Предсказание цен на квартиры в Москве.
**Алгоритм:** Random Forest Regression.
**Гиперпараметры:**
* Количество деревьев (n_estimators)
* Максимальная глубина дерева (max_depth)
**Метрики качества:**
* MSE: {mse:.2f}
* R²: {r2:.2f}
""".format(mse=mse, r2=r2))

# Построение графика
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, label="Предсказания")
plt.plot([0, 5], [0, 5], color='red', linewidth=2, label="Идеальное предсказание")
plt.xlabel("Истинные значения")
plt.ylabel("Предсказанные значения")
plt.title(f"Random Forest Regression (деревьев={n_estimators}, глубина={max_depth})")
plt.legend()
st.pyplot(plt)
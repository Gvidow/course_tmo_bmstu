import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error


@st.cache_data
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    return data

# # Загрузка данных
data = load_data()
# X = data.drop(columns=['Diabetes_binary'])
# y = data['Diabetes_binary']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

data_out = data.copy()
# Масштабирование числовых признаков
scaler = MinMaxScaler()
numeric_cols = data_out.select_dtypes(include=np.number).columns.tolist()
data_out[numeric_cols] = scaler.fit_transform(data_out[numeric_cols])

# Разделение на обучающую и тестовую выборки
X = data_out.drop(columns=['Diabetes_binary'])  # Замените 'target_column' на название вашего столбца с целевой переменной
y = data_out['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Функция для обучения и оценки модели
def train_and_evaluate(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return {
            "Точность выбранной модели (accuracy)": accuracy,
            "ROC AUC": roc_auc,
            "Среднеквадратичная ошибка (MSE)": mse,
            "Средняя абсолютная ошибка (MAE)": mae,
        }

# Заголовок приложения
st.title('Выбор модели и гиперпараметров')

# Выбор модели
model_name = st.sidebar.selectbox(
    'Выберите модель', 
    ('Метод опорных векторов', 'Случайный лес', 'Градиентный бустинг', 'Метод ближайших соседей')
)

if model_name == 'Метод опорных векторов':
    st.sidebar.subheader('Гиперпараметры метода опорных векторов')
    C = st.sidebar.slider('C (Параметр регуляризации)', 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox('Ядро', ('linear', 'poly', 'rbf', 'sigmoid'))
    gamma = st.sidebar.selectbox('Гамма', ('scale', 'auto'))
    model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    
elif model_name == 'Случайный лес':
    st.sidebar.subheader('Гиперпараметры случайного леса')
    n_estimators = st.sidebar.slider('Количество деревьев', 10, 200, 100)
    max_depth = st.sidebar.slider('Максимальная глубина', 1, 50, 10)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
elif model_name == 'Градиентный бустинг':
    st.sidebar.subheader('Гиперпараметры градиентного бустинга')
    learning_rate = st.sidebar.slider('Скорость обучения', 0.01, 1.0, 0.1)
    n_estimators = st.sidebar.slider('Количество деревьев', 10, 200, 100)
    max_depth = st.sidebar.slider('Максимальная глубина', 1, 50, 3)
    model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)

elif model_name == 'Метод ближайших соседей':
    st.sidebar.subheader('Гиперпараметры метода ближайших соседей')
    n_neighbors = st.sidebar.slider('Число соседей (k)', 1, 20, 5)
    weights = st.sidebar.selectbox('Весовые функции', ('uniform', 'distance'))
    algorithm = st.sidebar.selectbox('Алгоритм', ('auto', 'ball_tree', 'kd_tree', 'brute'))
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

# Обучение и оценка модели
# accuracy = train_and_evaluate(model)
stat = train_and_evaluate(model)
for title, metric in stat.items():
    st.write(f'{title}: {metric:.2f}')
    # st.write(f'Точность выбранной модели: {accuracy:.2f}')

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# функция построения модели линейной регрессии
def lin_reg (x_train, x_test, y_train,  y_test):
    # создание модели линейной регрессии вида mx+b
    reg = LinearRegression()
    # обучение модели линейной регрессии
    reg.fit(x_train, y_train)
    print('==========================Lin reg===========================')
    # вывод параметров модели (коэффициенты уравнения m (reg.coef_) и b(reg.intercept_))
    print(reg.coef_ , reg.intercept_)
    # подстановка значений обучающей выборки в уравнение (общая площадь из x_train)
    y_pred1 = reg.predict(x_train)
    # подстановка значений тестовой выборки в уравнение (общая площадь из x_test)
    y_pred2 = reg.predict(x_test)
    # оценка качества уравнения с помощью коэффициента детерминации R**2
    print(r2_score(y_train, y_pred1), r2_score(y_test, y_pred2))
    # оценка точности модели с помощью средней относительной ошибки
    print(mean_absolute_percentage_error(y_test, y_pred2))
    return y_pred2

# функция построения модели ближайшего соседа
def n_neighbor (x_train, x_test, y_train,  y_test):
    # создание модели ближайшего соседа с количеством соседей 7
    reg = KNeighborsRegressor(n_neighbors=7)
    # обучение модели ближайшего соседа
    reg.fit(x_train, y_train)
    # прогнозирование с использованием тестовой выборки и обученной модели
    y_pred = reg.predict(x_test)
    # оценка качества модели (расчет коэффициента детерминации)
    print(r2_score( y_test, y_pred))
    # оценка точности модели (средняя относительная ошибка)
    print(mean_absolute_percentage_error(y_test, y_pred))
    return y_pred

# функция построения модели дерево решений
def dec_tree (x_train, x_test, y_train,  y_test):
    # создаение дерева решений с параметрами 
    # min_samples_split - минимальное количество примеров в узле
    # минимальное количество примеров в листе
    reg = DecisionTreeRegressor(min_samples_split=5, min_samples_leaf=5)
    # обучение модели на обучающей выборке
    reg.fit(x_train, y_train)
    # прогнозирование с использованием тестовой выборки и обученной модели
    y_pred = reg.predict(x_test)
    # оценка качества модели (расчет коэффициента детерминации)
    print(r2_score( y_test, y_pred))
    # оценка точности модели (средняя относительная ошибка)
    print(mean_absolute_percentage_error(y_test, y_pred))
    return y_pred

# функция построения модели случайного леса
def random_forest (x_train, x_test, y_train,  y_test):
    # создание модели случайного леса с параметрами
    # min_samples_split - минимальное количество примеров в узле
    # минимальное количество примеров в листе
    reg = RandomForestRegressor(min_samples_split=6, min_samples_leaf=6)
    # обучение модели на обучающей выборке
    reg.fit(x_train, y_train)
    # прогнозирование с использованием тестовой выборки и обученной модели
    y_pred = reg.predict(x_test)
    # оценка качества модели (расчет коэффициента детерминации)
    print(r2_score( y_test, y_pred))
    # оценка точности модели (средняя относительная ошибка)
    print(mean_absolute_percentage_error(y_test, y_pred))
    return y_pred

# функция построения модели нейронной сети (многослойный перспетрон)
def NN (x_train, x_test, y_train,  y_test):
    # создание модели нейронной сети (многослойный перспетрон)
    reg = MLPRegressor(hidden_layer_sizes=(15,), batch_size=5, max_iter = 1000)
    # обучение модели на обучающей выборке
    reg.fit(x_train, y_train)
    # прогнозирование с использованием тестовой выборки и обученной модели
    y_pred = reg.predict(x_test)
    # оценка качества модели (расчет коэффициента детерминации)
    print(r2_score( y_test, y_pred))
    # оценка точности модели (средняя относительная ошибка)
    print(mean_absolute_percentage_error(y_test, y_pred))
    return y_pred

# кодирование категориальных признаков
def encode (df, column):
    # создание модели для кодирования данных целыми числами
    trans_data = LabelEncoder()
    # обучение модели кодирования
    trans_data.fit(df[column])
    # трансформирование исходных данных (замена строк на целые числа)
    df[column] = trans_data.transform(df[column])
    return df

# нормализация    
def norm (df, column):
    # создание модели нормализации (приведение к интервалу от 0 до 1)
    trans_data = MinMaxScaler()
    # обучение модели
    trans_data.fit(df[column])
    # трансформирование исходных данных (приведение к интервалу от 0 до 1)    
    df[column] = trans_data.transform(df[column])
    return df

# чтение данных
pd.set_option('display.max_columns', None)
df = pd.read_csv('19_Data.csv', sep=',')
 
# вывод информации о датафрейме
print(df.info())
print(df.shape)
print('============================================================')
print(df.head())
print('============================================================')
print(df.tail())
print('============================================================')

# вывод описательной статистики по первоначальному датасету
print(round(df.describe(include = "all"),2))
print('============================================================')

# подготовка данных
# преобразование в формат DateTime
df['DATOP'] = pd.to_datetime(df['DATOP'], format="%Y-%m-%d")
# создание и заполнение столбца с днем недели
df['day_week'] = df['DATOP'].dt.weekday
# перемещение столбца day_week в начало датафрейма
df = df[['day_week'] + [x for x in df.columns if x != 'day_week']]
# удаление столбцов
df.drop(columns=['ID', 'DATOP', 'STD', 'STA', 'STATUS'], inplace = True, axis = 1)
# кодирование категориальных признаков
for column in ['FLTID', 'DEPSTN', 'ARRSTN', 'AC']:
    df_tr = encode(df, column)
# очистка данных от значений превышающих 3 стандартных отклонения
df_tr = df_tr[df_tr['target']<df_tr['target'].mean()+df_tr['target'].std()*3]
# нормализация значений
df_tr_nrm = norm(df_tr, ['target'])

# вывод информации о датафрейме
print(df_tr_nrm.info())
print(df_tr_nrm.shape)
print('============================================================')
print(df_tr_nrm.head())
print('============================================================')
print(df_tr_nrm.tail())
print('============================================================')
# вывод описательной статистики по обработанному датасету
print(round(df_tr_nrm.describe(include = "all"),2))
result = round(df_tr_nrm.describe(include = "all"),2)
# result.to_excel('result.xlsx')
print('============================================================')

# построение корреляции
sns.heatmap(round(df_tr_nrm.corr(),2), annot=True, cbar=False)
plt.show()
print(df_tr_nrm.corr()['target'])

plt.figure(figsize = (16, 12))
plt.figure()
sns.heatmap(df_tr_nrm.corr(), annot = True , cmap = 'coolwarm', cbar=False)
plt.show()

# построение графиков зависимостей
fig = plt.figure(figsize = (16, 12))
for index, item in enumerate(df_tr_nrm.columns[:-1], start = 1):
    s = fig.add_subplot(3,2,index)
    s.set_title(item)
    plt.scatter(df_tr_nrm[item], df_tr_nrm['target'])
plt.show()

plt.figure(figsize = (16, 12))
plt.plot(df_tr_nrm["day_week"], df_tr_nrm["target"], color="blue", label="delay")
plt.grid()
plt.title("delay")
plt.legend()
plt.show()	

# разделение исходных данных на обучающую и тестовую выборки
x_train, x_test, y_train,  y_test = train_test_split(df_tr_nrm[df_tr_nrm.columns[:-1]],
                                                      df_tr_nrm['target'], train_size=0.9, random_state=1)
# моделирование
y1 = lin_reg(x_train, x_test, y_train,  y_test)

print('==========================Nearest_neighbor==================')
y2 = n_neighbor(x_train, x_test, y_train,  y_test)

print('==========================Decision tree======================')
y3 = dec_tree(x_train, x_test, y_train,  y_test)

print('==========================Random forest======================')
y4 = random_forest(x_train, x_test, y_train,  y_test)

print('==========================Neural network======================')
y5 = NN(x_train, x_test, y_train,  y_test)



from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math 
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, max_error, r2_score
import statsmodels.api as sm
import numpy as np
from scipy import stats
import pandas as pd


def train_test(data: pd.DataFrame, target_name: str):

    """Retornamos los datos escalados y splitados para el entrenamiento del modelo y tambien para los analisis 
    del performance del modelo"""

    # Preparacion de los datos para el modelo
    X_cols = list(set(data.columns) - set([target_name]))
    Y_cols = target_name

    X = data[X_cols].values
    Y = data[Y_cols].values.reshape(-1,1)

    # Split de datos
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.2, random_state= 42)

    # Escaladores para los datos
    sc_x = StandardScaler().fit(X)
    sc_y = StandardScaler().fit(Y)

    # Datos de entrenamiento y testeo escalados
    x_train = sc_x.transform(x_train)
    x_test = sc_x.transform(x_test)
    y_train = sc_y.transform(y_train)
    y_test = sc_y.transform(y_test)

    return x_train, x_test, y_train, y_test, sc_x, sc_y


def trained_model(x_train, y_train):

    """Modelo entrenamo con los datos splitados
    x_train: variables independientes para el entrenamiento
    y_train: variable dependiente para el entrenamiento
    intecept: Variable bool referida a si se desea crear o no el coef de intercepto en x = 0
    
    """
    # Modelo
    model = LinearRegression()
    return model.fit(x_train, y_train)


def analysis_test(x_test, y_test, model_x, sc_y_model):
    
    """ Funcion para evaluar el modelo con los datos de testeo
    Obtenemos: 
    - R-cuadrado
    - MAPE: Error absoluto medio porcentual
    - RMSE: Raiz de la media de errores al cuadrado
    - MMAX_ERR: Error maximo de prediccion"""

    model = model_x         # Modelo entrenado
    sc_y = sc_y_model       # Escalador entrenado 

    y_pred_std = model.predict(x_test)   # Predicciones

    # Transformamos las predicciones a escala original
    y_pred = sc_y.inverse_transform(y_pred_std)
    y_original = sc_y.inverse_transform(y_test)

    
    # Calcular R^2
    r2 = r2_score(y_original, y_pred)

    # Calcular otras métricas
    mape = mean_absolute_percentage_error(y_original, y_pred)
    rmse = math.sqrt(mean_squared_error(y_original, y_pred))
    max_err = max_error(y_original, y_pred)

    return round(r2,3), round(mape,3), round(rmse,3), round(max_err,3)
    
def analysis_train(x_train, y_train, df:pd.DataFrame, target_name:str):
    

    columns = list(set(df.columns) - set(target_name))

    model = sm.OLS(y_train, x_train)
    results = model.fit()

    # Extraer los valores deseados y redondear a 3 decimales
    r_squared = round(results.rsquared, 3)
    adj_r_squared = round(results.rsquared_adj, 3)
    f_statistic = round(results.fvalue, 3)
    prob_f_statistic = round(results.f_pvalue, 3)
    coefficients = results.params.round(4)
    p_values = results.pvalues.round(4)

    # Calcular el test de Omnibus, skew y kurtosis y redondear a 3 decimales
    omnibus_test = stats.normaltest(results.resid)
    omnibus = round(omnibus_test.statistic, 3)
    prob_omnibus = round(omnibus_test.pvalue, 3)
    skew = round(stats.skew(results.resid), 3)
    kurtosis = round(stats.kurtosis(results.resid), 3)

    # Calcular Durbin-Watson y redondear a 3 decimales
    durbin_watson = round(sm.stats.durbin_watson(results.resid), 3)

    # Crear la lista o diccionario con los datos solicitados
    resultados = {
        'R-squared': r_squared,
        'Adj. R-squared': adj_r_squared,
        'F-statistic': f_statistic,
        'Prob (F-statistic)': prob_f_statistic,
        'Coefficients and P-values': dict(zip(columns, zip(coefficients, p_values))),
        'Omnibus': omnibus,
        'Prob(Omnibus)': prob_omnibus,
        'Skew': skew,
        'Kurtosis': kurtosis,
        'Durbin-Watson': durbin_watson
    }

    return resultados
    
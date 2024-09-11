from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, max_error, r2_score
from scipy import stats
import math 
import statsmodels.api as sm
import numpy as np
import pandas as pd
import re 
import os 
import joblib

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

    results = {'R-squared_test': [round(r2,4)], 
               'MAPE_test':[round(mape,3)],
               'RMSE_test':[round(rmse,3)],
               'MAX_ERR_test':[round(max_err,3)]
               }

    return results
    
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
        'Coefficients and P-values': [dict(zip(columns, zip(coefficients, p_values)))],
        'Omnibus': omnibus,
        'Prob(Omnibus)': prob_omnibus,
        'Skew': skew,
        'Kurtosis': kurtosis,
        'Durbin-Watson': durbin_watson
    }

    return resultados



def save_model(model,
               analysis_train: dict,
               analysis_test: dict,
               X_cols: list, 
               target_name:str, 
               comentario_bitacora = None):
    """Esta funcion se encarga de guardar el modelo creado, y ademas almacenar en una bitacora
    los nombres de las columnas que se usaron para el modelo. Por ende es necesario ser especifico
    al momento de crear el nombre de las columnas ya que estas nos daran pistas sobre si sufrieron 
    alguna transformacion respecto a las columnas del dataframe original"""

    # Variables independientes y dependiente
    X_cols = X_cols
    Y_cols = target_name


    # Almacenado del modelo entrenado

    archivos = os.listdir('./modelos_entrenados')  

    # Patron de busqueda de coindidencias para saber cuandos modelos tenemos ya creados
    patron = r'.*\.pkl$'
    num = 0 # Numero de modelos creados

    for models in archivos:
        if re.search(patron, models):
            num += 1        # Asignacion del numero correspondiente al modelo

    name_model = f'model_{num}'

    joblib.dump(model, f'modelos_entrenados/{name_model}.pkl')   # Almacenamos el modelo
    # ----------------------------------------------------------------------------------

    # Bitacora de las columnas para el modelo
    with open('./modelos_entrenados/bitacora.txt','a', encoding='utf-8') as bitacora:
        
        contenido = f""" 
        ----------------{name_model}------------------
        - columns: 
        {X_cols}

        - Target: {Y_cols}

        - Comentario:
        {comentario_bitacora}

        """

        bitacora.write(contenido)
    # ----------------------------------------------------------------------------------

    # Almacenamiento de metricas de evaluacion

    df_analysis_train = pd.DataFrame(analysis_train)
    df_analysis_test = pd.DataFrame(analysis_test)

    # Extraer las primeras 3 columnas de df1
    df1_part1 = df_analysis_train.iloc[:, :4]
    # Extraer las columnas restantes de df1
    df1_part2 = df_analysis_train.iloc[:, 4:]


    df_name_model = pd.DataFrame({'Num_model':[name_model]})


    # Concatenar las primeras 3 columnas de df1, seguido de df2, y luego las columnas restantes de df1
    df_final = pd.concat([df_name_model,df1_part1, df_analysis_test, df1_part2], axis=1)

    if 'metricas_evaluacion.csv' in archivos:
        # print('archivo encontrado')

        df_existente = pd.read_csv('./modelos_entrenados/metricas_evaluacion.csv')
        df_concatenado = pd.concat([df_existente, df_final], ignore_index=True)

        # Guardar el DataFrame resultante en el archivo CSV (sobrescribiendo el archivo original)
        df_concatenado.to_csv('./modelos_entrenados/metricas_evaluacion.csv', index=False)

    else:
        # Exportar a CSV
        df_final.to_csv('./modelos_entrenados/metricas_evaluacion.csv', index=False)  # index=False evita que se guarde el índice
        # print('csv creado')

    # ----------------------------------------------------------------
    pd.set_option('display.max_colwidth', None) # Necesario para poder visualizar todo el contenido de la columna de coeficientes
    return df_final
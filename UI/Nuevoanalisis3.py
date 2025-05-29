# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:37:01 2024

@author: sigal
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import butter, filtfilt, lfilter, iirnotch
import os
import re


def prepros(datos, window_size=240, target_class="HablaImaginada"):
    mi = datos[:16, :].astype(np.float64)
    etiquetas_clase = datos[16, :]
    etiquetas_marca = datos[17, :]
    fs = 125  # Frecuencia de muestreo en Hz
    # Cortador de la señal ################################################
    def cortador(s, labels):
        s_cut = s[:, :np.where(labels == "u")[0][-1]+1]
        return s_cut

    # Separación de un canal ##############################################
    senal = cortador(mi, etiquetas_clase)
  

    # Separación de un canal ##############################################

    # Re-referenciación ###################################################
    def rereference(data):
        """
        Aplica referencia común restando la media de todos los canales a cada canal.
        """
        return data - np.mean(data, axis=0, keepdims=True)

    senal = rereference(senal)

    # Detrend de la señal #################################################
    def highpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    cutoff = 0.1  # Frecuencia de corte
    senal_detrend = highpass_filter(senal, cutoff, fs)

    # Filtro pasa banda ###################################################
    def bandpass_filter(data, lowcut, highcut, fs, order=8):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    senal_beta = bandpass_filter(senal_detrend, 8, 62, fs)
    def notch_filter(data, freq, fs, quality=30):
        """
        Aplica un filtro notch (de rechazo de banda) para eliminar una frecuencia específica.

        Parámetros:
        - data: señal de entrada.
        - freq: frecuencia a eliminar (por defecto 60 Hz).
        - fs: frecuencia de muestreo.
        - quality: factor de calidad del filtro (afecta el ancho de la banda de rechazo).

        Retorna:
        - Señal filtrada.
        """
        nyquist = 0.5 * fs
        w0 = freq / nyquist  # Normalización de la frecuencia
        b, a = iirnotch(w0, quality)
        return filtfilt(b, a, data)

    # Aplicación del filtro notch a 60 Hz
    senal_filtrada = notch_filter(senal_beta, 60, fs)
    senal_filtrada = notch_filter(senal_filtrada, 50, fs)
    # Normalización de la señal ###########################################
    scaler_std = StandardScaler()
    senal_normalized = scaler_std.fit_transform(senal_filtrada.T).T
    
    datos_hi = datos[:, :senal_normalized.shape[1]]
    senal_procesada = np.vstack((senal_normalized, datos_hi[16:, :]))
    return (senal_procesada)



# Función para separar los segmentos por clase y fase
def separar_segmentos_por_clase_y_fase(arr):
    """
    Separa los segmentos de datos según la clase y la fase.
    Los segmentos de la misma fase se agrupan juntos.

    Parámetros:
        - arr: Arreglo de forma (18, n) donde las filas 17 y 18 contienen
               la clase y la fase, respectivamente.

    Retorna:
        - Un diccionario donde las claves son las fases y dentro de cada fase
          hay otro diccionario con las clases como claves y los segmentos como valores.
    """
    # Obtener las clases y fases
    clases = arr[16, :]  # La clase está en la fila 17 (índice 16)
    fases = arr[17, :]   # La fase está en la fila 18 (índice 17)

    # Crear un diccionario para almacenar los resultados
    segmentos = {}

    # Variables para hacer el seguimiento de los segmentos
    segmento_actual = None
    fase_anterior = None
    clase_anterior = None

    # Iterar sobre los datos y separar por clase y fase
    for i in range(arr.shape[1]):  # Iterar por cada columna (cada segmento)
        clase = clases[i]
        fase = fases[i]

        # Si estamos en una nueva fase, y no es la misma fase que la anterior, agregamos el segmento anterior
        if fase != fase_anterior or clase != clase_anterior:
            # Guardar el segmento actual si existe
            if segmento_actual is not None:
                if fase_anterior not in segmentos:
                    segmentos[fase_anterior] = {}

                if clase_anterior not in segmentos[fase_anterior]:
                    segmentos[fase_anterior][clase_anterior] = []

                segmentos[fase_anterior][clase_anterior].append(
                    segmento_actual)

            # Comenzar un nuevo segmento
            # Tomamos los 16 canales de cada segmento (sin la fila 17 y 18)
            segmento_actual = arr[:16, i]
            fase_anterior = fase
            clase_anterior = clase
        else:
            # Si la fase y clase son las mismas, agregamos el dato al segmento actual
            segmento_actual = np.column_stack((segmento_actual, arr[:16, i]))

    # Al final del ciclo, agregamos el último segmento
    if segmento_actual is not None:
        if fase_anterior not in segmentos:
            segmentos[fase_anterior] = {}

        if clase_anterior not in segmentos[fase_anterior]:
            segmentos[fase_anterior][clase_anterior] = []

        segmentos[fase_anterior][clase_anterior].append(segmento_actual)

    return segmentos


def ajustar_tamano_segmentos(segmentos):
    """
    Ajusta el tamaño de los segmentos dentro de cada clase y fase para que todos tengan
    el mismo número de columnas, el cual será el menor número de columnas de los segmentos
    en esa clase.

    Parámetros:
        - segmentos: Diccionario que contiene los segmentos organizados por fase y clase.

    Retorna:
        - Un diccionario con los segmentos ajustados a tamaño uniforme.
    """
    segmentos_ajustados = {}

    # Iterar sobre cada fase y clase
    for fase, clases in segmentos.items():
        if fase not in segmentos_ajustados:
            segmentos_ajustados[fase] = {}

        for clase, segs in clases.items():
            # Encontrar el número mínimo de columnas entre los segmentos de esa clase
            min_columnas = min(seg.shape[1] for seg in segs)
            if (np.abs(360-min_columnas)<=np.abs(240-min_columnas)):
                ajuste=360
            else:
                ajuste=240
            #Ajustar tamaño de columnas
            # Ajustar todos los segmentos a este tamaño
            segmentos_ajustados[fase][clase] = []

            for seg in segs:
                # Recortar o ajustar el segmento
                if seg.shape[1] > ajuste:
                    # Recortar el segmento si tiene más columnas que el mínimo
                    segmento_ajustado = seg[:, :ajuste]
                elif seg.shape[1] < ajuste:
                    # Rellenar el segmento si tiene menos columnas que el mínimo
                    # Aquí lo estamos rellenando con ceros (puedes cambiarlo si necesitas otro valor)
                    segmento_ajustado = np.hstack([seg, np.zeros((seg.shape[0], ajuste - seg.shape[1]))])
                else:
                    segmento_ajustado = seg
                segmentos_ajustados[fase][clase].append(np.array(segmento_ajustado).astype(np.float64))

    return segmentos_ajustados


"""
datos = np.load('1-LCQB-1729186040897.npz')
datos_hi=datos['Data']
senal_procesada=prepros(datos_hi)
# Separar los segmentos
segmentos = separar_segmentos_por_clase_y_fase(senal_procesada)

segmentos_ajustados = ajustar_tamano_segmentos(segmentos)
"""
# Obtener la ruta del directorio donde se encuentra el archivo actual
ruta_actual = os.path.dirname(os.path.realpath(__file__))

# Obtener todos los archivos en el directorio
archivos = [f for f in os.listdir(
    ruta_actual) if os.path.isfile(os.path.join(ruta_actual, f))]
vpalabras = [
    "aceptar",
    "cancelar",
    "arriba",
    "abajo",
    "derecha",
    "izquierda",
    "hola",
    "ayuda",
    "gracias",
    "a",
    "e",
    "i",
    "o",
    "u"
]
HI = {
    'aceptar': [],
    'cancelar': [],
    'arriba': [],
    'abajo': [],
    'derecha': [],
    'izquierda': [],
    'hola': [],
    'ayuda': [],
    'gracias': [],
    'a': [],
    'e': [],
    'i': [],
    'o': [],
    'u': []
}
HA = {
    'aceptar': [],
    'cancelar': [],
    'arriba': [],
    'abajo': [],
    'derecha': [],
    'izquierda': [],
    'hola': [],
    'ayuda': [],
    'gracias': [],
    'a': [],
    'e': [],
    'i': [],
    'o': [],
    'u': []
}
EA  = {
    'aceptar': [],
    'cancelar': [],
    'arriba': [],
    'abajo': [],
    'derecha': [],
    'izquierda': [],
    'hola': [],
    'ayuda': [],
    'gracias': [],
    'a': [],
    'e': [],
    'i': [],
    'o': [],
    'u': []
}
EV = {
    'aceptar': [],
    'cancelar': [],
    'arriba': [],
    'abajo': [],
    'derecha': [],
    'izquierda': [],
    'hola': [],
    'ayuda': [],
    'gracias': [],
    'a': [],
    'e': [],
    'i': [],
    'o': [],
    'u': []
}
EO = {
    'aceptar': [],
    'cancelar': [],
    'arriba': [],
    'abajo': [],
    'derecha': [],
    'izquierda': [],
    'hola': [],
    'ayuda': [],
    'gracias': [],
    'a': [],
    'e': [],
    'i': [],
    'o': [],
    'u': []
}
RL= {
    'aceptar': [],
    'cancelar': [],
    'arriba': [],
    'abajo': [],
    'derecha': [],
    'izquierda': [],
    'hola': [],
    'ayuda': [],
    'gracias': [],
    'a': [],
    'e': [],
    'i': [],
    'o': [],
    'u': []
}
# Mostrar los nombres de los archivos
arch_rep = np.zeros((len(archivos)-3))


for i in range(0,43):
    if  re.search(r'-(.*?)-', archivos[i]).group(1) == "AM":
        print(archivos[i])
        datos = np.load(archivos[i])
        datos_hi = datos['Data']
        senal_procesada = (prepros(datos_hi))
        # Separar los segmentos
        segmentos = separar_segmentos_por_clase_y_fase(senal_procesada)
        segmentos_ajustados = ajustar_tamano_segmentos(segmentos)
        for j in range(0, len(vpalabras)):
            HI[vpalabras[j]].extend(segmentos_ajustados["HablaImaginada"][vpalabras[j]])
            HA[vpalabras[j]].extend(segmentos_ajustados["HablaAbierta"][vpalabras[j]])
            EV[vpalabras[j]].extend(segmentos_ajustados["Palabra"][vpalabras[j]])
            EA[vpalabras[j]].extend(segmentos_ajustados["PalabraAudio"][vpalabras[j]])
            EO[vpalabras[j]].extend(segmentos_ajustados["PalabraAbierta"][vpalabras[j]])
            RL[vpalabras[j]].extend(segmentos_ajustados["Relajacion"][vpalabras[j]])
    

vpa=[key for key in HI]
valoreshi = np.array([HI[key] for key in HI])
valoresha = np.array([HA[key] for key in HA])
valoresev = np.array([EV[key] for key in EV])
valoresea = np.array([EA[key] for key in EA])
valoreseo = np.array([EO[key] for key in EO])
valoresrl = np.array([RL[key] for key in RL])



np.savez('s-2-prepros-ASU-50.npz',labels=vpa,hi=valoreshi, ha=valoresha, ea=valoresea,
         ev=valoresev, eo=valoreseo, rl=valoresrl)

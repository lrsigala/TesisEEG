# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:31:59 2023

@author: sigal
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from PyQt5 import uic
import sys
import os
import logging
import argparse

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import numpy as np
import pandas as pd
import time

objfases = {
    "Relajacion":0,
    "Actividad":1,
    "PalabraAudio":2,
    "IndicadorX":3,
    "PalabraAbierta":4,
    "Palabra":5,
    "HablaImaginada":6,
    "HablaAbierta":6
}

vtiempos = [3, 3, 2, 2, 2, 2, 2]
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
vfases = [
    "Relajacion",
    "PalabraAudio",
    "HablaImaginada",
    "Palabra",
    "HablaImaginada",
    "PalabraAbierta",
    "HablaAbierta",
]


class open_Subject(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("PantallaSujeto.ui", self)
        self.stackw = self.findChild(QStackedWidget, "stackedWidget")
        self.label_6= self.findChild(QLabel, "label_6")
        self.label_8= self.findChild(QLabel, "label_8")
        self.label_10= self.findChild(QLabel, "label_10")



class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        uic.loadUi("PantallaInicial.ui", self)

        self.bCerrar = self.findChild(QPushButton, "bCerrar")
        self.bIniciar = self.findChild(
            QPushButton, "bIniciar")  # Iniciar Prueba
        self.lTempo = self.findChild(QLabel, "lTempo")

        self.bSesion = self.findChild(QPushButton, "bSesion")

        self.bPausa = self.findChild(QPushButton, "bPausa")
        self.bSujeto = self.findChild(QPushButton, "bSujeto")

        self.tIniciales = self.findChild(QLineEdit, "tIniciales")
        self.tNumero = self.findChild(QLineEdit, "tNumero")
        self.bGuardar = self.findChild(QPushButton, "bGuardar")

        self.tNotas = self.findChild(QTextEdit, "tNotas")

        self.Gwidget = self.findChild(QWidget, "widget")

        self.bSujeto.clicked.connect(self.open_Subject)
        self.bSesion.clicked.connect(self.IniciarSesion)
        self.bIniciar.clicked.connect(self.IniciarPrueba)  # Iniciar Prueba
        self.bCerrar.clicked.connect(self.CloseButton)
        self.bPausa.clicked.connect(self.PausaPrueba)
        self.bGuardar.clicked.connect(self.GuardarSesion)

        self.bPausa.setVisible(False)
        self.bIniciar.setVisible(False)

        self.etapa = [False, False]
        self.recording = False
        self.record = []
        self.prompts=[]
        self.state=[]
        self.elapsed_time = 0
        self.index = 0
        self.indexPalabra = 0

        self.player = QMediaPlayer()
        self.content = []

        for i in range(0, len(vpalabras)):
            audio="Audios/"+vpalabras[i]+".mp3"
            full_file_path = os.path.join(os.getcwd(),audio)
            url = QUrl.fromLocalFile(full_file_path)
            self.content.append(url)

        self.UiComponents()

        self.show()

    def __ready(self):
        if False not in self.etapa:
            self.bPausa.setVisible(True)
            self.bIniciar.setVisible(True)
        else:
            self.bPausa.setVisible(False)
            self.bIniciar.setVisible(False)

    def open_Subject(self):
        self.etapa[0] = True
        self.__ready()
        self.w = open_Subject()
        self.w.show()

    def UiComponents(self):
        layout = QVBoxLayout()
##################################### Grafica ###############################################

        graphWidget = pg.GraphicsLayoutWidget()
        self.x = list()  #
        self.y = list(np.zeros(500))  # 100 data points
        for i in range(0, 16):
            self.x.append([0 for _ in range(500)])
            self.record.append([])

        graphWidget.setBackground('w')
        colors = ['black', 'purple', 'blue', 'green',
                  '#a89d32', 'orange', 'red', 'brown']

        self.curves = list()
        h = 0
        for i in range(0, 2):
            for j in range(0, 8):
                p = graphWidget.addPlot(row=j, col=i)
                p.showAxis('bottom', False)
                p.showAxis('left', False)
                pen = pg.mkPen(color=(255, 0, 0))
                # p.plot(self.x,self.y,pen=colors[j])
                curve = p.plot(pen=colors[j])
                self.curves.append(curve)
                self.curves[h].setData(self.x[h])
                h = h+1
        layout.addWidget(graphWidget)
        self.Gwidget.setLayout(layout)

    def IniciarSesion(self, checked):
        if self.etapa[1] == False:
            self.timer = QTimer()
            self.timer.timeout.connect(self.updategraph)
            self.sampling_rate = board_shim.get_sampling_rate(
                board_shim.board_id)

            self.update_speed_ms = 4 #Ajustar a la frecuencia de muestreo
            self.window_size = 8
            self.num_points = self.window_size * self.sampling_rate
            self.timer.start(self.update_speed_ms)
            self.etapa[1] = True
            self.__ready()
            self.bSesion.setText('Detener Transmisión')
        else:
            self.timer.stop()
            self.etapa[1] = False
            self.__ready()
            self.bSesion.setText('Iniciar Transmisión')

    def updategraph(self):
        data = board_shim.get_current_board_data(self.num_points)
        for i in range(0, len(self.curves)):
            self.x[i] = self.x[i][self.num_points:]  # Remove the first
            DataFilter.detrend(data[i+1], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[i+1],  # Datos por filtrar
                                        self.sampling_rate,  # fs Frecuencia de muestreo
                                        3.0,  # f1
                                        45.0,  # f2
                                        2,  # Orden del filtro
                                        FilterTypes.BUTTERWORTH.value,  # Tipo del filtro
                                        0)  # Ripple value para Chebyshev
            DataFilter.perform_bandstop(data[i+1], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[i+1],  self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)

            self.x[i].extend(data[i+1])
            self.curves[i].setData(self.x[i])
            if(self.recording == True):
                self.record[i].extend(data[i+1][-1:])
                if(i==0):
                    self.prompts.extend(np.full(len(data[i+1][-1:]), vpalabras[self.indexPalabra]))
                    self.state.extend(np.full(len(data[i+1][-1:]), vfases[self.index]))

###################################### Cambios Pantalla Sujeto ###############################################

    def IniciarPrueba(self):
        if self.bIniciar.text() == 'Iniciar Prueba':
            self.w.stackw.setCurrentIndex(objfases[vfases[0]])
            self.w.label_6.setText(vpalabras[0])
            self.w.label_8.setText(vpalabras[0])
            self.w.label_10.setText(vpalabras[0])
            self.timerP = QTimer()
            self.timerP.setInterval(vtiempos[0]*1000)
            self.timerP.timeout.connect(self.update)
            self.timerP.start()
            self.recording = True
            self.bIniciar.setText('Terminar Prueba')
            self.timerT = QTimer()
            self.timerT.timeout.connect(self.update_timer)
            self.timerT.start(1000)

    def update_timer(self):
        self.elapsed_time += 1
        self.update_display()

    def update_display(self):
        seconds = self.elapsed_time % 60
        minutes = (self.elapsed_time // 60) % 60
        #hours = self.elapsed_time // 3600

        time_str = '{:02d}:{:02d}'.format(minutes, seconds)
        self.lTempo.setText(time_str)

    def PausaPrueba(self):
        if self.recording == True:
            self.bPausa.setText('Continuar Prueba')
            self.timerP.stop()
            self.timerT.stop()
            self.recording = False
        else:
            self.bPausa.setText('Pausar Prueba')
            self.timerP.start()
            self.timerT.start()

            self.recording = True

    def update(self):
        self.index = self.index+1
        if (self.index == len(vfases)):
            self.index = 0
            self.indexPalabra = self.indexPalabra +1
            if (self.indexPalabra == len(vpalabras)):
                self.indexPalabra=0   
            self.w.label_6.setText(vpalabras[self.indexPalabra])
            self.w.label_8.setText(vpalabras[self.indexPalabra])
            self.w.label_10.setText(vpalabras[self.indexPalabra])
        if (self.index == 1):
            self.playsound()
        self.w.stackw.setCurrentIndex(objfases[vfases[self.index]])
        
        self.timerP.setInterval(vtiempos[self.index]*1000)

    def playsound(self):
        self.player.setMedia(QMediaContent(self.content[self.indexPalabra]))
        self.player.play()
#####################################################################################

    def GuardarSesion(self):
        a = ['','','']
        a[0] = self.tIniciales.text()
        a[1] = self.tNumero.text()
        a[2] = self.tNotas.toPlainText()
        
        recordnp = np.array(self.record)
        data=np.vstack((recordnp,np.transpose(self.prompts),np.transpose(self.state)))
        
        filename = a[1]+'-'+a[0]+'-'+str(int(time.time()*1000))+'.npz'
        np.savez(filename, Desc=a, Data=data)
        
    def CloseButton(self):
        self.timerP.stop()
        self.timerT.stop()
        self.timer.stop()

        self.close()
        qApp.quit()


BoardShim.enable_dev_board_logger()

parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str,
                    help='serial port', required=False, default='')
parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                    required=False, default=BoardIds.SYNTHETIC_BOARD)
args = parser.parse_args()

params = BrainFlowInputParams()
params.serial_port = args.serial_port


try:
    board_shim = BoardShim(args.board_id, params)
    board_shim.prepare_session()
    board_shim.start_stream()
    app = QApplication(sys.argv)

    UIWindow = UI()
    UIWindow.show()

    app.exec()

finally:
    if board_shim.is_prepared():
        board_shim.release_session()

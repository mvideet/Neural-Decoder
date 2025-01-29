print("Before PyQt imports...")
from PyQt5 import QtWidgets
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

print("After PyQt imports...")

app = QtWidgets.QApplication([])
w = QtWidgets.QWidget()
w.setWindowTitle("Hello PyQt")
w.show()

print("Before exec_...")
app.exec_()
print("After exec_ (this prints when the window is closed).")

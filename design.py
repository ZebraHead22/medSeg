# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(985, 753)
        MainWindow.setMinimumSize(QtCore.QSize(763, 678))
        MainWindow.setStyleSheet("background-color: rgb(45, 50, 80);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("QMainWindow {\n"
"    background-color: rgb(230, 230, 230);\n"
"    }")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setMinimumSize(QtCore.QSize(0, 298))
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget_5 = QtWidgets.QWidget(self.widget_2)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_12.setContentsMargins(12, 0, 12, 0)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.raw_image = QtWidgets.QLabel(self.widget_5)
        self.raw_image.setMinimumSize(QtCore.QSize(298, 298))
        self.raw_image.setMaximumSize(QtCore.QSize(400, 400))
        self.raw_image.setStyleSheet("QLabel {\n"
"    border-radius: 5px;\n"
"}")
        self.raw_image.setText("")
        self.raw_image.setScaledContents(True)
        self.raw_image.setObjectName("raw_image")
        self.horizontalLayout_12.addWidget(self.raw_image)
        self.horizontalLayout.addWidget(self.widget_5)
        self.widget_6 = QtWidgets.QWidget(self.widget_2)
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout_13.setContentsMargins(12, 0, 12, 0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.mask_l = QtWidgets.QLabel(self.widget_6)
        self.mask_l.setMinimumSize(QtCore.QSize(298, 298))
        self.mask_l.setMaximumSize(QtCore.QSize(400, 400))
        self.mask_l.setStyleSheet("QLabel {\n"
"    border-radius: 5px;\n"
"}")
        self.mask_l.setText("")
        self.mask_l.setScaledContents(True)
        self.mask_l.setObjectName("mask_l")
        self.horizontalLayout_13.addWidget(self.mask_l)
        self.horizontalLayout.addWidget(self.widget_6)
        self.widget_7 = QtWidgets.QWidget(self.widget_2)
        self.widget_7.setObjectName("widget_7")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.widget_7)
        self.horizontalLayout_14.setContentsMargins(12, 0, 12, 0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.segm_l = QtWidgets.QLabel(self.widget_7)
        self.segm_l.setMinimumSize(QtCore.QSize(298, 298))
        self.segm_l.setMaximumSize(QtCore.QSize(400, 400))
        self.segm_l.setStyleSheet("QLabel {\n"
"    border-radius: 5px;\n"
"}")
        self.segm_l.setText("")
        self.segm_l.setScaledContents(True)
        self.segm_l.setObjectName("segm_l")
        self.horizontalLayout_14.addWidget(self.segm_l)
        self.horizontalLayout.addWidget(self.widget_7)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.widget_3 = QtWidgets.QWidget(self.widget)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.widget_8 = QtWidgets.QWidget(self.widget_3)
        self.widget_8.setMinimumSize(QtCore.QSize(440, 0))
        self.widget_8.setObjectName("widget_8")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_8)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_10 = QtWidgets.QWidget(self.widget_8)
        self.widget_10.setObjectName("widget_10")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget_10)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.widget_13 = QtWidgets.QWidget(self.widget_10)
        self.widget_13.setObjectName("widget_13")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_13)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label = QtWidgets.QLabel(self.widget_13)
        self.label.setMinimumSize(QtCore.QSize(0, 33))
        self.label.setMaximumSize(QtCore.QSize(416, 33))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color: rgb(66, 71, 105);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_5.addWidget(self.label)
        self.verticalLayout_4.addWidget(self.widget_13)
        self.widget_14 = QtWidgets.QWidget(self.widget_10)
        self.widget_14.setObjectName("widget_14")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_14)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.widget_14)
        self.label_2.setMinimumSize(QtCore.QSize(0, 33))
        self.label_2.setMaximumSize(QtCore.QSize(50, 33))
        self.label_2.setStyleSheet("background-color: rgb(66, 71, 105);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.x_loc_label = QtWidgets.QLabel(self.widget_14)
        self.x_loc_label.setMinimumSize(QtCore.QSize(0, 33))
        self.x_loc_label.setMaximumSize(QtCore.QSize(16777215, 33))
        self.x_loc_label.setStyleSheet("background-color: rgb(95, 101, 151);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.x_loc_label.setText("")
        self.x_loc_label.setAlignment(QtCore.Qt.AlignCenter)
        self.x_loc_label.setObjectName("x_loc_label")
        self.horizontalLayout_4.addWidget(self.x_loc_label)
        self.label_4 = QtWidgets.QLabel(self.widget_14)
        self.label_4.setMinimumSize(QtCore.QSize(0, 33))
        self.label_4.setMaximumSize(QtCore.QSize(50, 33))
        self.label_4.setStyleSheet("background-color: rgb(66, 71, 105);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.y_loc_label = QtWidgets.QLabel(self.widget_14)
        self.y_loc_label.setMinimumSize(QtCore.QSize(0, 33))
        self.y_loc_label.setMaximumSize(QtCore.QSize(16777215, 33))
        self.y_loc_label.setStyleSheet("background-color: rgb(95, 101, 151);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.y_loc_label.setText("")
        self.y_loc_label.setAlignment(QtCore.Qt.AlignCenter)
        self.y_loc_label.setObjectName("y_loc_label")
        self.horizontalLayout_4.addWidget(self.y_loc_label)
        self.verticalLayout_4.addWidget(self.widget_14)
        self.verticalLayout_3.addWidget(self.widget_10)
        self.widget_11 = QtWidgets.QWidget(self.widget_8)
        self.widget_11.setObjectName("widget_11")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget_11)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.widget_15 = QtWidgets.QWidget(self.widget_11)
        self.widget_15.setObjectName("widget_15")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_15)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_8 = QtWidgets.QLabel(self.widget_15)
        self.label_8.setMinimumSize(QtCore.QSize(0, 33))
        self.label_8.setMaximumSize(QtCore.QSize(416, 33))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("background-color: rgb(66, 71, 105);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_5.addWidget(self.label_8)
        self.verticalLayout_6.addWidget(self.widget_15)
        self.widget_16 = QtWidgets.QWidget(self.widget_11)
        self.widget_16.setObjectName("widget_16")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget_16)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_11 = QtWidgets.QLabel(self.widget_16)
        self.label_11.setMinimumSize(QtCore.QSize(0, 33))
        self.label_11.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_11.setStyleSheet("background-color: rgb(66, 71, 105);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_6.addWidget(self.label_11)
        self.kras_label = QtWidgets.QLabel(self.widget_16)
        self.kras_label.setMinimumSize(QtCore.QSize(0, 33))
        self.kras_label.setStyleSheet("background-color: rgb(95, 101, 151);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.kras_label.setText("")
        self.kras_label.setAlignment(QtCore.Qt.AlignCenter)
        self.kras_label.setObjectName("kras_label")
        self.horizontalLayout_6.addWidget(self.kras_label)
        self.label_13 = QtWidgets.QLabel(self.widget_16)
        self.label_13.setMinimumSize(QtCore.QSize(0, 33))
        self.label_13.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_13.setStyleSheet("background-color: rgb(66, 71, 105);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_6.addWidget(self.label_13)
        self.egfr_label = QtWidgets.QLabel(self.widget_16)
        self.egfr_label.setMinimumSize(QtCore.QSize(0, 33))
        self.egfr_label.setStyleSheet("background-color: rgb(95, 101, 151);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.egfr_label.setText("")
        self.egfr_label.setAlignment(QtCore.Qt.AlignCenter)
        self.egfr_label.setObjectName("egfr_label")
        self.horizontalLayout_6.addWidget(self.egfr_label)
        self.verticalLayout_6.addWidget(self.widget_16)
        self.verticalLayout_3.addWidget(self.widget_11)
        self.widget_12 = QtWidgets.QWidget(self.widget_8)
        self.widget_12.setObjectName("widget_12")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.widget_12)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.term_wi = QtWidgets.QWidget(self.widget_12)
        self.term_wi.setObjectName("term_wi")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.term_wi)
        self.verticalLayout_13.setContentsMargins(-1, 0, -1, 0)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.label_3 = QtWidgets.QLabel(self.term_wi)
        font = QtGui.QFont()
        font.setFamily(".SF NS")
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("QLabel {\n"
"    border-radius: 5px;\n"
"    background-color: rgb(0, 0, 0);\n"
"    color: white;\n"
"}")
        self.label_3.setText("")
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_13.addWidget(self.label_3)
        self.verticalLayout_7.addWidget(self.term_wi)
        self.verticalLayout_3.addWidget(self.widget_12)
        self.horizontalLayout_3.addWidget(self.widget_8)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.widget_9 = QtWidgets.QWidget(self.widget_3)
        self.widget_9.setObjectName("widget_9")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget_9)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.widget_21 = QtWidgets.QWidget(self.widget_9)
        self.widget_21.setObjectName("widget_21")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.widget_21)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.widget_23 = QtWidgets.QWidget(self.widget_21)
        self.widget_23.setMaximumSize(QtCore.QSize(16777215, 33))
        self.widget_23.setObjectName("widget_23")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.widget_23)
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.line_getcwd = QtWidgets.QLineEdit(self.widget_23)
        self.line_getcwd.setMinimumSize(QtCore.QSize(0, 33))
        self.line_getcwd.setMaximumSize(QtCore.QSize(16777215, 33))
        self.line_getcwd.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.line_getcwd.setStyleSheet("background-color: rgb(95, 101, 151);\n"
"border-radius: 5px;\n"
"color: white;\n"
"")
        self.line_getcwd.setMaxLength(500)
        self.line_getcwd.setCursorPosition(0)
        self.line_getcwd.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.line_getcwd.setObjectName("line_getcwd")
        self.verticalLayout_10.addWidget(self.line_getcwd)
        self.verticalLayout_9.addWidget(self.widget_23)
        self.widget_19 = QtWidgets.QWidget(self.widget_21)
        self.widget_19.setObjectName("widget_19")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.widget_19)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.widget_25 = QtWidgets.QWidget(self.widget_19)
        self.widget_25.setObjectName("widget_25")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.widget_25)
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.listWidget = QtWidgets.QListWidget(self.widget_25)
        self.listWidget.setMinimumSize(QtCore.QSize(0, 330))
        self.listWidget.setMaximumSize(QtCore.QSize(16777215, 350))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.listWidget.setFont(font)
        self.listWidget.setStyleSheet("background-color: rgb(95, 101, 151);\n"
"border-radius: 5px;\n"
"color: white;\n"
"\n"
"")
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_11.addWidget(self.listWidget)
        self.horizontalLayout_9.addWidget(self.widget_25)
        self.widget_24 = QtWidgets.QWidget(self.widget_19)
        self.widget_24.setMinimumSize(QtCore.QSize(170, 0))
        self.widget_24.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.widget_24.setAutoFillBackground(False)
        self.widget_24.setObjectName("widget_24")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.widget_24)
        self.verticalLayout_12.setContentsMargins(12, 0, 0, 0)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.comboBox = QtWidgets.QComboBox(self.widget_24)
        self.comboBox.setMinimumSize(QtCore.QSize(176, 33))
        self.comboBox.setMaximumSize(QtCore.QSize(120, 30))
        self.comboBox.setStyleSheet("background-color: rgb(66, 71, 105);\n"
"border-radius: 5px;\n"
"padding-left: 5px;\n"
"\n"
"QComboBox::drop-down {\n"
"    width: 0px;\n"
"    height: 0px;\n"
"    border: 0px;\n"
"}\n"
"\n"
"QComboBox QAbstractItemView {\n"
"    color: rgb(255, 255, 255);    \n"
"    border-radius: 5px;\n"
"    background-color: rgb(95, 101, 151);\n"
"    padding-left: 5px;\n"
"    selection-background-color: rgb(39, 44, 54);\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"    margin-left: 10px;\n"
"}\n"
"")
        self.comboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.comboBox.setMinimumContentsLength(10)
        self.comboBox.setIconSize(QtCore.QSize(5, 5))
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout_12.addWidget(self.comboBox)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_12.addItem(spacerItem1)
        self.browse_btn = QtWidgets.QPushButton(self.widget_24)
        self.browse_btn.setMinimumSize(QtCore.QSize(176, 30))
        self.browse_btn.setMaximumSize(QtCore.QSize(176, 33))
        self.browse_btn.setStyleSheet("QPushButton {\n"
"    border-radius: 5px;\n"
"    background-color: rgba(246, 173, 113, 203);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    border-radius: 5px;\n"
"    background-color: rgb(246, 173, 113);\n"
"    color: white;\n"
"}")
        self.browse_btn.setObjectName("browse_btn")
        self.verticalLayout_12.addWidget(self.browse_btn)
        self.process_btn = QtWidgets.QPushButton(self.widget_24)
        self.process_btn.setMinimumSize(QtCore.QSize(176, 30))
        self.process_btn.setMaximumSize(QtCore.QSize(176, 33))
        self.process_btn.setStyleSheet("QPushButton {\n"
"    border-radius: 5px;\n"
"    background-color: rgba(246, 173, 113, 203);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    border-radius: 5px;\n"
"    background-color: rgb(246, 173, 113);\n"
"    color: white;\n"
"}")
        self.process_btn.setObjectName("process_btn")
        self.verticalLayout_12.addWidget(self.process_btn)
        self.ext_btn = QtWidgets.QPushButton(self.widget_24)
        self.ext_btn.setMinimumSize(QtCore.QSize(176, 30))
        self.ext_btn.setMaximumSize(QtCore.QSize(176, 33))
        self.ext_btn.setStyleSheet("QPushButton {\n"
"    border-radius: 5px;\n"
"    background-color: rgba(246, 173, 113, 203);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    border-radius: 5px;\n"
"    background-color: rgb(246, 173, 113);\n"
"    color: white;\n"
"}")
        self.ext_btn.setObjectName("ext_btn")
        self.verticalLayout_12.addWidget(self.ext_btn)
        self.horizontalLayout_9.addWidget(self.widget_24)
        self.verticalLayout_9.addWidget(self.widget_19)
        self.verticalLayout_8.addWidget(self.widget_21)
        self.horizontalLayout_3.addWidget(self.widget_9)
        self.verticalLayout_2.addWidget(self.widget_3)
        self.verticalLayout.addWidget(self.widget)
        self.widget_4 = QtWidgets.QWidget(self.centralwidget)
        self.widget_4.setMaximumSize(QtCore.QSize(16777215, 80))
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout.addWidget(self.widget_4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setStyleSheet("QStatusBar {\n"
"    color:white;\n"
"}")
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "medSeg v.0.0.1"))
        self.label.setText(_translate("MainWindow", "Coordination of Nodule"))
        self.label_2.setText(_translate("MainWindow", "  X:"))
        self.label_4.setText(_translate("MainWindow", "  Y:"))
        self.label_8.setText(_translate("MainWindow", "Mutation Possibility"))
        self.label_11.setText(_translate("MainWindow", " KRAS:"))
        self.label_13.setText(_translate("MainWindow", " EGFR:"))
        self.browse_btn.setText(_translate("MainWindow", "Browse"))
        self.process_btn.setText(_translate("MainWindow", "Process"))
        self.ext_btn.setText(_translate("MainWindow", "Exit"))

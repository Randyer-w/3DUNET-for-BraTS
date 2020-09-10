import os,sys
from PyQt5 import QtCore,QtWidgets,QtGui
from train import config
from show import *
class test():

    def setUI(self,w):
        #设置工具窗口的大小
        w.setGeometry(600,600,600,400)
        #设置工具窗口的标题
        w.setWindowTitle("医学图像分割程序")
        QtWidgets.QToolTip.setFont(QtGui.QFont('SansSerif',10))

        self.label = QtWidgets.QLabel(w)
        self.label.setGeometry(QtCore.QRect(50, 25, 500, 45))
        self.label.setFont(QtGui.QFont("Roman times",12))
        self.label.setText("将数据放入data/original/mydata目录下，并输入金标准的label值：")

        self.label2 = QtWidgets.QLabel(w)
        self.label2.setGeometry(QtCore.QRect(250, 250, 300, 100))
        self.label2.setFont(QtGui.QFont("Roman times",16,QtGui.QFont.Bold))

        self.label3 = QtWidgets.QLabel(w)
        self.label3.setGeometry(QtCore.QRect(250, 110, 300, 100))
        self.label3.setFont(QtGui.QFont("Roman times", 16, QtGui.QFont.Bold))

        #添加设置一个文本框
        self.text = QtWidgets.QLineEdit(w)
        #调整文本框的位置大小
        self.text.setGeometry(QtCore.QRect(220,100,160,30))

        self.btn_label = QtWidgets.QPushButton(w)
        self.btn_label.move(400, 100)
        self.btn_label.setText("确定")
        self.btn_label.clicked.connect(self.inputlabel)  # 为按钮添加单击事件

        #/////////////////////////////////第一行
        self.btn_zipdata = QtWidgets.QPushButton(w)
        self.btn_zipdata.move(40,250)
        self.btn_zipdata.setText("开始预处理")
        self.btn_zipdata.clicked.connect(self.run_zipdata)#为按钮添加单击事件

        self.btn_train = QtWidgets.QPushButton(w)
        self.btn_train.move(150, 250)
        self.btn_train.setText("训练")
        self.btn_train.clicked.connect(self.run_train)

        self.btn_predict = QtWidgets.QPushButton(w)
        self.btn_predict.move(260, 250)
        self.btn_predict.setText("预测")
        self.btn_predict.clicked.connect(self.run_predict)

        self.btn_showloss = QtWidgets.QPushButton(w)
        self.btn_showloss.move(370, 250)
        self.btn_showloss.setText("训练的loss曲线")
        self.btn_showloss.clicked.connect(self.run_showloss)

        self.btn_shownii = QtWidgets.QPushButton(w)
        self.btn_shownii.move(480, 250)
        self.btn_shownii.setText("预测结果显示")
        self.btn_shownii.clicked.connect(self.run_shownii)
        # /////////////////////////////////第二行
        self.btn_showunet = QtWidgets.QPushButton(w)
        self.btn_showunet.move(40, 350)
        self.btn_showunet.setText("显示Unet网络结构")
        self.btn_showunet.clicked.connect(self.run_showunet)

        self.btn_showdicebox = QtWidgets.QPushButton(w)
        self.btn_showdicebox.move(150, 350)
        self.btn_showdicebox.setText("显示DICE系数箱形图")
        self.btn_showdicebox.clicked.connect(self.run_showdicebox)

        w.show()

    def run_predict(self):
        self.label3.setText("")
        self.label2.setText("正在预测")
        os.system("sh ./predict.sh")
        self.label2.setText("预测完成")

    def run_train(self):
        self.label3.setText("")
        self.label2.setText("正在训练")
        os.system("sh ./train.sh")
        self.label2.setText("训练完成")

    def run_zipdata(self):
        self.label3.setText("")
        name = self.text.text()
        self.label2.setText("正在预处理：%s" % name)
        #os.system("sh ./zipdata.sh")
        self.label2.setText("预处理完成")

    def run_shownii(self):
        show_nii();

    def run_showloss(self):
        show_loss();

    def inputlabel(self):
        name = self.text.text()
        if name:
            if(name=="1"):
                config["labels"] = (1,)
            elif(name=="1,2,4"):
                config["labels"] = (1,2,4)
            self.label3.setText("设置成功")
        else :
            self.label3.setText("设置失败")
        print(config["labels"])

    def run_showdicebox(self):
        show_dicebox()

    def run_showunet(self):
        run_showunet()

if __name__=='__main__':
    #创建应用程序和对象
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QWidget()
    ui = test()
    ui.setUI(w)
    sys.exit(app.exec_())
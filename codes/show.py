from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import os
from PyQt5.QtWidgets import QLabel
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

def show_loss():
    try:
        img=Image.open('./loss_graph.png')
    except:
        os.system("sh ./evaluate.sh")
        try:
            img = Image.open('./loss_graph.png')
        except:
            print("生成失败，训练时发生错误，请重新训练！！")
            print("程序已关闭，请重新运行")
            exit(0)
    plt.figure("show")
    plt.imshow(img)
    plt.show()

def show_nii():
    try:
        example_filename = './prediction/hyy/prediction.nii'
        img = nib.load(example_filename)
    except:
        os.system(" cd prediction/hyy && gunzip prediction.nii.gz")
        example_filename = './prediction/hyy/prediction.nii'
        img = nib.load(example_filename)
    width, height, queue = img.dataobj.shape
    OrthoSlicer3D(img.dataobj).show()
    num = 1
    for i in range(0, queue, 10):
        img_arr = img.dataobj[:, :, i]
        plt.subplot(5, 4, num)
        plt.imshow(img_arr, cmap='gray')
        num += 1
    plt.show()
    os.system(" cd prediction/hyy && gzip prediction.nii")

def show_dicebox():
    try:
        img=Image.open('./validation_scores_boxplot.png')
    except:
        #print("正在进行评估")
        os.system("sh ./evaluate.sh")
        img = Image.open('./validation_scores_boxplot.png')
    plt.figure("show")
    plt.imshow(img)
    plt.show()

def run_showunet():#展示unet结构
    img=Image.open('./3D-Unet.png')
    plt.figure("3D-Net结构展示")
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    show_loss()
    show_nii()
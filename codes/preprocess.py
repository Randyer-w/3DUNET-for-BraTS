"""
用于转换，规范化和矫正数据。
"""


import glob
import os
import warnings
import shutil

import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection

from train import config


def append_basename(in_file, append):
    dirname, basename = os.path.split(in_file)
    base, ext = basename.split(".", 1)
    return os.path.join(dirname, base + append + "." + ext)


def get_background_mask(in_folder, out_file, truth_name="GlistrBoost_ManuallyCorrected"):#获取所有数据的背景mask
    #in_folder：输入数的文件夹。
    #out_file：包含掩码为1的图像，其中该主题的图像数据包含背景。
    #truth_name：如何在主题文件夹中标记真值文件
    background_image = None
    for name in config["all_modalities"] + [truth_name]:
        image = sitk.ReadImage(get_image(in_folder, name))
        if background_image:
            if name == truth_name and not (image.GetOrigin() == background_image.GetOrigin()):
                image.SetOrigin(background_image.GetOrigin())
            background_image = sitk.And(image == 0, background_image)
        else:
            background_image = image == 0
    sitk.WriteImage(background_image, out_file)
    return os.path.abspath(out_file)


def convert_image_format(in_file, out_file):#转换图像格式
    sitk.WriteImage(sitk.ReadImage(in_file), out_file)
    return out_file


def window_intensities(in_file, out_file, min_percent=1, max_percent=99):
    image = sitk.ReadImage(in_file)
    image_data = sitk.GetArrayFromImage(image)
    out_image = sitk.IntensityWindowing(image, np.percentile(image_data, min_percent), np.percentile(image_data,
                                                                                                     max_percent))
    sitk.WriteImage(out_image, out_file)
    return os.path.abspath(out_file)


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):#偏执矫正 首先尝试使用ANT N4BiasFieldCorrection，若失败则尝试使用SimpleITK来纠正偏差
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection 矫正偏执失败"
                                     "正在尝试使用SimpleITK来纠正偏差"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)


def rescale(in_file, out_file, minimum=0, maximum=20000):
    image = sitk.ReadImage(in_file)
    sitk.WriteImage(sitk.RescaleIntensity(image, minimum, maximum), out_file)
    return os.path.abspath(out_file)


def get_image(subject_folder, name):
    file_card = os.path.join(subject_folder, "*" + name + ".nii.gz")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))


def background_to_zero(in_file, background_file, out_file):
    sitk.WriteImage(sitk.Mask(sitk.ReadImage(in_file), sitk.ReadImage(background_file, sitk.sitkUInt8) == 0),
                    out_file)
    return out_file


def check_origin(in_file, in_file2):
    image = sitk.ReadImage(in_file)
    image2 = sitk.ReadImage(in_file2)
    if not image.GetOrigin() == image2.GetOrigin():
        image.SetOrigin(image2.GetOrigin())
        sitk.WriteImage(image, in_file)


def normalize_image(in_file, out_file, bias_correction=True):
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file


def convert_brats_folder(in_folder, out_folder, truth_name='seg', no_bias_correction_modalities=None):
    for name in config["all_modalities"]:
        try:
            image_file = get_image(in_folder, name)
        except RuntimeError as error:
            if name == 't1ce':
                image_file = get_image(in_folder, 't1Gd')
                truth_name = "GlistrBoost_ManuallyCorrected"
            else:
                raise error

        out_file = os.path.abspath(os.path.join(out_folder, name + ".nii.gz"))
        perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
        normalize_image(image_file, out_file, bias_correction=perform_bias_correction)
    # 复制groundtruth文件
    try:
        truth_file = get_image(in_folder, truth_name)
    except RuntimeError:
        truth_file = get_image(in_folder, truth_name.split("_")[0])

    out_file = os.path.abspath(os.path.join(out_folder, "truth.nii.gz"))
    shutil.copy(truth_file, out_file)
    check_origin(out_file, get_image(in_folder, config["all_modalities"][0]))


def convert_brats_data(brats_folder, out_folder, overwrite=False, no_bias_correction_modalities=("flair",)):
    #no_bias_correction_modalities: 执行偏差校正可以减少某些模式的信号。如果有关的特定模态信号被矫正了减少，可以通过来指定其不被矫正。
    for subject_folder in glob.glob(os.path.join(brats_folder, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                convert_brats_folder(subject_folder, new_subject_folder,
                                     no_bias_correction_modalities=no_bias_correction_modalities)

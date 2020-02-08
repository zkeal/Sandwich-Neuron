# -*-coding:utf-8-*-

import sys
import getopt
import matplotlib.pylab as plt
import numpy as np
from skimage import io,filters,feature,morphology, measure, segmentation

from Data_prepare import processor
from Image_scaner import Scanner as IS
import os as sysos
import warnings

warnings.filterwarnings('ignore')





if __name__ == '__main__':

    options, args = getopt.getopt(sys.argv[1:], "h:s:l:u:e:m:", ["help", "section_scale=", "lower_threshold=",
                                "upper_threshold=", "erosion=","median="])

    # default value
    section_scale = 2
    lower_threshold = 0.6
    upper_threshold = 1.0
    erosion = 2
    median = 5

    for option, value in options:
        # if option in ("-h", "--help"):
        if option in ("-s", "--section_scale"):
            section_scale = float(value)
        if option in ("-l", "--lower_threshold"):
            lower_threshold = float(value)
        if option in ("-u", "--upper_threshold"):
            upper_threshold = float(value)
        if option in ("-e", "--erosion"):
            erosion = float(erosion)
        if option in ("-m", "--median"):
            median = float(median)

    Data_p = processor(section_scale,lower_threshold,upper_threshold,erosion,median)


    sample_file = "labeled_image"
    samp_filename = sample_file+"/" + sysos.listdir(sample_file)[0]
    label_name = "large_high_pixel_labels"
    group_labels, var, average_width = Data_p.residual_groups(samp_filename)
    final_coefficent, y_min_residual, final_curve_x, final_curve_y, function_curve_x, function_curve_y = Data_p.residual_variance(
        group_labels)

    scan = IS()
    scan.__int__(average_width, final_coefficent, 0.1, y_min_residual)
    average_width, average_hight, average_thresh, extract_label, average_hu, daisy = Data_p.overall_label_information(
        label_name, samp_filename)


    file_list = sysos.listdir("images")
    print("Start Processing Images...\n")
    for filename in file_list:
        print("\n Processing images: ",filename)
        im = io.imread("images/"+filename)
        im = im.T

        # canvas = np.load("results/example_image.tif.npy")
        # RESULT = Data_p.overall_quantification(im, canvas, average_thresh, daisy, filename)

        feature_point = scan.scan(im, final_curve_x, final_curve_y, average_width, average_hight,
                                  average_thresh)
        section_list = Data_p.mark_up(im, feature_point, round(average_width * 0.8), round(average_hight * 0.8),
                               average_hu)
        canvas = Data_p.combine_section(section_list, im.shape[0], im.shape[1])
        RESULT = Data_p.overall_quantification(im, canvas, average_thresh, daisy,filename)
        np.save("results/" + filename + str(RESULT) + ".npy", canvas)



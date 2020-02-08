import numpy as np
import B_spline_curve as B
import matplotlib.pylab as plt
import warnings
import progressbar

class Scanner():
    def __int__(self, width, coefficient_group, noisy_tolerance,y_min_residual):
        self.width = round(width)
        self.coefficient_group = coefficient_group
        self.noisy_tolerance = noisy_tolerance
        self.y_min_residual = y_min_residual
        self.feature_points = []

    def scan(self, image,curve_x, curve_y, average_width, average_hight , average_thresh,ruler = 10):
        line_index =0
        min_range = np.sqrt(np.square(average_width) + np.square(average_hight))
        try:
            with progressbar.ProgressBar(max_value=image.shape[0]) as bar:
                print('Start scanning image for distribution function...')
                while line_index < image.shape[0]:
                    check_index = 0
                    while check_index < image.shape[1]:
                        check_index = check_index + 1
                        theorical_curve_x = []
                        theorical_curve_y = []
                        intensity = np.mean(image[line_index][check_index:check_index + self.width])
                        if intensity < average_thresh/2:
                            continue
                        data_section = B.batch_normalization(image[line_index][check_index:check_index + self.width],ruler)
                        for i_index in range(0, len(data_section) - 2):
                            iimage_coefficent = B.two_spline_curve(range(0, len(data_section))[i_index:i_index + 3], data_section[i_index:i_index + 3])
                            section_plot = B.plot_2curve(iimage_coefficent)
                            theorical_curve_x.extend(section_plot[2])
                            theorical_curve_y.extend(section_plot[3])
                        y_residual_variance = 0
                        if len(data_section) < ruler:
                            continue
                        for curve_index in range(0, len(curve_x)):
                            for theorical_curve_index_x in range(0,len(theorical_curve_x)):
                                if np.abs(curve_x[curve_index] - theorical_curve_x[theorical_curve_index_x]) < 0.001:
                                    y_residual_variance = y_residual_variance + np.square(curve_y[curve_index] - theorical_curve_y[theorical_curve_index_x])
                        if (1 + self.noisy_tolerance) * self.y_min_residual > y_residual_variance:
                            self.aggregate_group((line_index,check_index),min_range/4)
                            check_index = check_index + round(average_hight)
                            check_index = check_index + 1
                    line_index = line_index + 1
                    bar.update(line_index)
        except Warning:
            print('Warning was raised as an exception!')
        except TypeError:
            print(check_index)



                # for curve_index in range(0, len(curve_x)):
                #     y_residual_variance = y_residual_variance + np.square(data_section[curve_x[curve_index]] - curve_y[curve_index])
                # y_residual_variance = y_residual_variance/len(curve_x)


        return self.feature_points


    def aggregate_group(self, feature_point,range):

        if len(feature_point) != 2:
            return
        if len(self.feature_points) == 0:
            self.feature_points.append(feature_point)
            return
        for t_points in self.feature_points:
            now_range = np.sqrt(
                np.square(np.abs(t_points[0] - feature_point[0])) + np.square(np.abs(t_points[1] - feature_point[1])))
            if np.sqrt(np.square(np.abs(t_points[0] - feature_point[0])) + np.square(np.abs(t_points[1] - feature_point[1]))) < range:
                return
        self.feature_points.append(feature_point)
        #print(feature_point[0], feature_point[1])

    def auto_width_detection(self,image,index,threshold,average_label_width,average_label_hight):
        width_index = 0
        ight_index = 0

        select_width =0

        while width_index < image.shape[0]:
            select_width =0
            connected_component_flag = 0
            while select_width < average_label_width and connected_component_flag !=0:
                width_index = width_index + 1





# -*-coding:utf-8-*-

import matplotlib.pylab as plt
import numpy as np
from skimage.util import img_as_float
from skimage import io,filters,feature,morphology, measure, segmentation
from skimage.measure import label, regionprops,moments_hu
import cv2
import B_spline_curve as B
import progressbar

class processor(object):
    def __init__(self,section_scale,lower_threshold, upper_threshold,erosion,median):
        self.section_scale = section_scale
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.erosion = erosion
        self.median = median



    def read_labels(self,label_file):
        Labels = []
        f = open(label_file, "r+")
        for line in f:
            axis = line.rstrip("\n").split(",")
            Labels.append(list(map(int, axis)))
        return Labels

    def cut_labels(self,labels, image):
        for label in labels:
            extract_area = image[label[0]:label[2], label[1]:label[3]]
            var = []
            label_values = []
            for line in extract_area:
                var.append(np.var(line))
                label_values.append(line)
            yield label_values, var

    def plti(self,im, **kwargs, ):
        plt.imshow(im, interpolation="none", **kwargs)
        plt.axis('off')  # 去掉坐标轴
        plt.show()  # 弹窗显示图像

    def residual_groups(self,file_name, ruler=10):
        index_counter = 0
        sum_width = 0
        temprary_var = []
        temprary_label_values = {}
        im = io.imread(file_name)
        im = im.T
        labels = self.read_labels(label_file="large_high_pixel_labels")
        for label_values, var in self.cut_labels(labels, im):
            sum_width = sum_width + np.array(label_values).shape[1]
            if index_counter == 0:
                for index in range(0, len(label_values)):
                    temprary_label_values[index] = B.batch_normalization(label_values[index], ruler)
                temprary_var = var
                index_counter = index_counter + 1
                continue
            min_range = 0x7fffffff
            # result_mean = []
            # mean_now = 0

            for index_var1 in range(0, len(temprary_var)):
                minmize_index = 0
                for index_var2 in range(0, len(var)):
                    var2_label = B.batch_normalization(label_values[index_var2], ruler)
                    curve_measurement = np.sum(np.abs(temprary_label_values[index_var1] - var2_label))
                    if curve_measurement < min_range:
                        min_range = curve_measurement
                        minmize_index = index_var2
                if len(temprary_label_values[index_var1].shape) != 1:
                    listtt = temprary_label_values[index_var1].tolist()
                    listtt.append(B.batch_normalization(label_values[minmize_index].tolist(), ruler).tolist())
                    temprary_label_values[index_var1] = np.array(listtt)
                else:
                    tem_value = temprary_label_values[index_var1].tolist()
                    sd = B.batch_normalization(label_values[minmize_index], ruler).tolist()
                    temprary_label_values[index_var1] = np.array([tem_value, sd])
                min_range = 0x7fffffff

            index_counter = index_counter + 1
        return temprary_label_values, temprary_var, float(sum_width / index_counter)

    def residual_variance(self,label_values):
        y_min_residual = 0x7fffffff
        final_coefficent = []
        final_curve_x = []
        final_curve_y = []
        function_curve_x = []
        function_curve_y = []

        min_key = 0

        for key in range(0, len(label_values)):
            y_residual = 0
            y_mean = []
            coefficents_group = []
            curve_x = []
            curve_y = []
            t_function_curve_x = []
            t_function_curve_y = []
            if len(label_values[key]) != 0:
                for i in range(0, label_values[key].shape[1]):
                    y_mean.append(np.mean(label_values[key][:, i]))
                for i_2 in range(0, len(y_mean) - 2):
                    coefficent = B.two_spline_curve(range(0, len(y_mean))[i_2:i_2 + 3], y_mean[i_2:i_2 + 3])
                    section_plot = B.plot_2curve(coefficent)
                    curve_x.extend(section_plot[0])
                    curve_y.extend(section_plot[1])
                    t_function_curve_x.extend(section_plot[2])
                    t_function_curve_y.extend(section_plot[3])
                    coefficents_group.append(coefficent)

                for i in range(0, label_values[key].shape[0]):
                    for compare_x_axis in range(0, len(curve_x)):
                        y_residual = y_residual + np.square(
                            label_values[key][i][curve_x[compare_x_axis]] - curve_y[compare_x_axis])

                y_residual = y_residual / label_values[key].shape[0]

            if y_residual < y_min_residual:
                y_min_residual = y_residual
                final_coefficent = coefficents_group
                final_curve_x = curve_x
                final_curve_y = curve_y
                function_curve_x = t_function_curve_x
                function_curve_y = t_function_curve_y
                min_key = key

        # plt.plot(final_curve_x, final_curve_y)
        # plt.show()

        # for test_curve in label_values[min_key]:
        #     plt.plot(range(0, len(test_curve)), test_curve)
        #     plt.show()

        return final_coefficent, y_min_residual, final_curve_x, final_curve_y, function_curve_x, function_curve_y

    def mark_up(self,ima, feature_point, average_width, average_height, average_hu):
        section_list = []
        process_bar = 0
        with progressbar.ProgressBar(max_value=len(feature_point)) as bar:
            print("Analyzing sections...")
            for point in feature_point:
                try:
                    lower_width = round(point[1] - average_width if point[1] - average_width > 0 else 0)
                    up_width = round(point[1] + round(average_width * self.section_scale) if point[1] + round(
                        average_width * self.section_scale) < ima.shape[1] else ima.shape[1])
                    lower_height = round(point[0] - average_height if point[0] - average_height > 0 else 0)
                    up_height = round(point[0] + round(average_height * self.section_scale) if point[0] + round(
                        average_height * self.section_scale) < ima.shape[0] else ima.shape[0])
                    extract_area = ima[lower_height:up_height, lower_width:up_width]
                    # plti(extract_area)
                    gray_scaleImage = self.find_section_connected_component(extract_area, average_width, average_height,
                                                                       average_hu)
                    section_list.append((gray_scaleImage, lower_width, up_width, lower_height, up_height))
                    # plti(gray_scaleImage)
                    process_bar = process_bar + 1
                    bar.update(process_bar)
                except ValueError as e:
                    print(point)
        return section_list

    def overall_label_information(self,label_file, image, ruler=10):
        im = io.imread(image)
        im = im.T
        labels = self.read_labels(label_file=label_file)
        average_width = 0
        average_hight = 0
        average_hu = np.zeros((7,))
        average_daisy = np.zeros((104,))
        thresh = 0
        extract_label = np.zeros((ruler, ruler))

        test_array = []
        for label in labels:
            T_threshold, dst, hu, daisy = self.grayscale_image(im, label)
            # if np.sum(average_daisy) != 0:
            #     print(np.mean(np.corrcoef(daisy[0][0], average_daisy)))
            average_daisy = daisy + average_daisy
            average_hu = average_hu + hu
            average_width = average_width + label[2] - label[0]
            average_hight = average_hight + label[3] - label[1]
            thresh = thresh + T_threshold
            extract_label = extract_label + dst
            test_array.append(dst)
        average_width = average_width / len(labels)
        average_hight = average_hight / len(labels)
        average_hu = average_hu / len(labels)
        average_daisy = average_daisy / len(labels)
        average_thresh = filters.threshold_otsu(im)
        extract_label = (extract_label > int(ruler * 1.4 / 2)) * 1.0
        # plti(extract_label, cmap='gray_r')
        # np.save('extrac_label.npy',extract_label)

        return average_width, average_hight, average_thresh, extract_label, average_hu, average_daisy

    def grayscale_image(self,im, label=None, ruler=10):
        compressed_image = []
        vertical_compressed = []
        if label is not None:

            extract_area = im[label[0]:label[2], label[1]:label[3]]

            for line_index in range(0, extract_area.shape[0]):
                compressed_image.append(B.batch_normalization(extract_area[line_index,], ruler))
            compressed_image = np.array(compressed_image).T
            for vertical_index in range(0, compressed_image.shape[0]):
                vertical_compressed.append(B.batch_normalization(compressed_image[vertical_index,], ruler))
            vertical_compressed = np.array(vertical_compressed).T
            T_threshold = filters.threshold_li(vertical_compressed)
            dst = (vertical_compressed >= T_threshold) * 1.0
            decs = self.get_daisy(extract_area)
            hu = moments_hu(extract_area)
            return T_threshold, dst, hu, decs
        else:
            extract_area = im
            for line_index in range(0, extract_area.shape[0]):
                compressed_image.append(B.batch_normalization(extract_area[line_index,], ruler))
            compressed_image = np.array(compressed_image).T
            for vertical_index in range(0, compressed_image.shape[0]):
                vertical_compressed.append(B.batch_normalization(compressed_image[vertical_index,], ruler))
            vertical_compressed = np.array(vertical_compressed).T
            T_threshold = filters.threshold_li(vertical_compressed)
            dst = (vertical_compressed >= T_threshold) * 1.0
            ##plti(dst, cmap='gray_r')
            return T_threshold, dst

    def combine_section(self,section_list, canvas_hight, canvas_width):
        background = np.zeros((canvas_hight, canvas_width))
        for section in section_list:
            gray_scaleImage, lower_width, up_width, lower_height, up_hight = section
            # if lower_height-up_hight != gray_scaleImage.shape[0] or lower_width-up_width != gray_scaleImage.shape[1]:
            #     continue
            gray_scaleImage = self.polrozation_image(gray_scaleImage)
            background[lower_height:up_hight, lower_width:up_width] = background[lower_height:up_hight,
                                                                      lower_width:up_width] + gray_scaleImage
        return background

    def get_daisy(self,extract_area):
        shrink = cv2.resize(extract_area, (extract_area.shape[1], extract_area.shape[1]), interpolation=cv2.INTER_AREA)
        descs = feature.daisy(shrink, step=extract_area.shape[1], radius=int(extract_area.shape[1] / 2) - 1, rings=2,
                              histograms=6,
                              orientations=8, visualize=False)

        x_norm = np.linalg.norm(descs[0][0])
        descs = descs / x_norm
        return descs[0][0]

        # shrink = cv2.resize(extract_area, (extract_area.shape[1], extract_area.shape[1]), interpolation=cv2.INTER_AREA)
        # descs,image = feature.daisy(shrink, step=extract_area.shape[1], radius=int(extract_area.shape[1] / 2) - 1, rings=2,
        #                       histograms=6,
        #                       orientations=8, visualize=True)
        # plti(image)
        # x_norm = np.linalg.norm(descs[0][0])
        # descs = descs / x_norm
        # return descs[0][0]

    def find_section_connected_component(self,extract_area, average_width, average_height, average_hu):
        # extract_area = np.load("70.npy")
        # plti(extract_area)
        list_threshold = []
        list_region_count = []
        gray_scale_imagelist = []

        threshold_li = int(filters.threshold_li(extract_area))
        median_image = filters.median(extract_area, morphology.disk(self.median))

        # binary_image = morphology.diameter_closing(median_image,23, connectivity=2)
        ##plti(binary_image, cmap = 'gray_r')

        edge_temp = feature.canny(img_as_float(extract_area), sigma=1.9)
        ##plti(edge_temp)
        edge_temp = morphology.binary_closing(edge_temp, morphology.disk(self.erosion))
        ##plti(edge_temp)
        edge_temp = morphology.erosion(~edge_temp * 1.0, morphology.disk(self.erosion))
        # plti(edge_temp)

        for T_threshold in range(int(threshold_li), int(threshold_li) * 3, int(threshold_li / 10)):

            binary_image = morphology.opening(median_image > T_threshold, morphology.disk(1))

            # edge_temp = feature.canny(img_as_float(extract_area),sigma= float(T_threshold/10))
            # edge_temp = ~edge_temp * 1.0
            # #plti(edge_temp, cmap='gray_r')

            label_image = label(binary_image * edge_temp)
            # plti(label_image)
            region_count = 0
            for region in regionprops(label_image):
                kk = np.mean(np.corrcoef(region.moments_hu, average_hu))
                if np.mean(np.corrcoef(region.moments_hu,
                                       average_hu)) > 0.3 and region.area > average_width * average_height * 0.2:
                    region_count = region_count + 1
                # take regions with large enough areas
                # if region.area >= average_width* average_height*0.2:
                #     region_count = region_count + 1
                # #draw rectangle around segmented coins
                # minr, minc, maxr, maxc = region.bbox
                # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                #                           fill=False, edgecolor='red', linewidth=2)
                # ax.add_patch(rect)

            list_threshold.append(T_threshold)
            list_region_count.append(region_count)
            gray_scale_imagelist.append(label_image)

        group_counts = np.max(list_region_count)
        list_region_count = list_region_count[::-1]
        selected_index = list_region_count.index(group_counts)
        # ax.set_axis_off()
        # plt.tight_layout()
        np.save("temp.npy", extract_area)
        gray_scale_imagelist = gray_scale_imagelist[::-1]
        # plti(gray_scale_imagelist[selected_index])
        return gray_scale_imagelist[selected_index]

    def polrozation_image(self,grey_scaleimage):
        grey_scaleimage = (grey_scaleimage >= 1) * 1.0
        # grey_scaleimage = (grey_scaleimage < 1) * -1.0
        return grey_scaleimage

    def overall_quantification(self,orginal_image, final_canvas, average_thresh, daisy,filename):

        hu = []
        # final_canvas = np.load('raw_background.npy')
        overall_thres = int(filters.threshold_otsu(final_canvas))

        overall_binary = (final_canvas > overall_thres) * 1

        #self.plti(overall_binary, cmap='gray_r')

        # overall_binary = morphology.erosion(overall_binary, morphology.disk(1))

        neurons_labels = label(overall_binary)



        #self.plti(neurons_labels)

        # extract_label = np.load('extrac_label.npy')

        region_count = len(regionprops(neurons_labels))

        for region in regionprops(neurons_labels):
            try:
                # plti(region.image, cmap = 'gray_r')
                minr, minc, maxr, maxc = region.bbox
                # if np.mean(np.corrcoef(region.moments_hu, hu)) > 0.4:
                #     rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red',
                #                              linewidth=2)
                #     ax.add_patch(rect)
                if region.area < 150:
                    overall_binary[minr:maxr, minc:maxc] = 0
                    region_count = region_count - 1
                    continue
                if np.mean(orginal_image[minr:maxr, minc:maxc]) < (average_thresh * self.lower_threshold):
                    overall_binary[minr:maxr, minc:maxc] = 0
                    region_count = region_count - 1
                    continue

                value = orginal_image[minr:maxr, minc:maxc]
                #self.plti(value)
                #print(value.shape)
                daisy_region = self.get_daisy(value)
                kkk = np.mean(np.corrcoef(daisy, daisy_region))
                # plti(orginal_image[minr:maxr, minc:maxc])
                # plti(overall_binary[minr:maxr, minc:maxc], cmap = 'gray_r')
                if np.mean(np.corrcoef(daisy, daisy_region)) < 0.5:
                    overall_binary[minr:maxr, minc:maxc] = 0
                    region_count = region_count - 1
            except ValueError:
                minr, minc, maxr, maxc = region.bbox
                err_location = str(minr)+","+str(minc)+","+str(maxr)+"," + str(maxc)
                print("error at" + err_location)
            # if region.area < 100:
            #     overall_binary[minr:maxr, minc:maxc] = 0\
        # test = segmentation.find_boundaries(overall_binary, mode='outer').astype(np.uint8)
        #
        # plti(test, cmap = 'gray')

        contours = measure.find_contours(overall_binary, 0.3)

        # Display the image and plot all contours found
        fig, ax = plt.subplots(figsize=(final_canvas.shape[1] / 100, final_canvas.shape[0] / 100))
        ax.set_facecolor('black')
        # ax.imshow(orginal_image)

        count = 0

        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            count = count + 1

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig("results/"+filename+".png")
        plt.show()

        return region_count



# def boundary_test(average_hu):
#
#     # raw_background = np.load('raw_background.npy')
#     # #plti(raw_background,cmap = 'gray_r')
#     #
#     #
#     # extract_label = np.load('extrac_label.npy')
#     # #plti(extract_label,cmap = 'gray_r')
#     #
#     # TT = int(filters.threshold_otsu(raw_background))
#     #
#     # raw_edge = (raw_background > TT) * 1.0
#     #
#     #
#     # plti(raw_edge,cmap = 'gray_r')
#     #
#     # aim_T = 0
#     # max_count=0
#     # max_threshold = 0
#     # for temp_T in range(TT, TT*3, int(TT/5)):
#     #     avail_count = 0
#     #     raw_edge = (raw_background > TT) * 1.0
#     #
#     #     label_image = label(raw_background)
#     #     for region in regionprops(label_image):
#     #         minr, minc, maxr, maxc = region.bbox
#     #         if region.area > 100 and region.area < 6000:
#     #             avail_count = avail_count + 1
#     #     if avail_count > max_count:
#     #         max_count = avail_count
#     #         max_threshold = temp_T
#     #
#     # raw_edge = (raw_background > max_threshold) * 1.0
#     #
#     # #plti(raw_edge,cmap = 'gray_r')
#     # dst1=morphology.erosion(raw_edge,morphology.square(2))
#     # #plti(dst1, cmap='gray_r')
#     # label_image = label(dst1)
#     #
#     # contours = measure.find_contours(dst1, 0.5)
#     #
#     # fig, axes = plt.subplots(1, 2, figsize=(raw_background.shape[0] / 100, raw_background.shape[1] / 100))
#     # ax0, ax1 = axes.ravel()
#     # ax0.imshow(raw_background, cmap='gray_r')
#     #
#     # rows, cols = raw_background.shape
#     # ax1.axis([0, rows, cols, 0])
#     # for n, contour in enumerate(contours):
#     #     ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
#     # ax1.axis('image')
#     # ax1.set_title('contours')
#     # plt.show()
#
#
#     # fig, ax = plt.subplots(figsize=(raw_background.shape[0] / 100, raw_background.shape[1] / 100))
#     # ax.imshow(test_data)
#     # for region in regionprops(label_image):
#     #     minr, minc, maxr, maxc = region.bbox
#     #     if maxr - minr > 10 and maxc - minc > 10:
#     #         region_data = grayscale_image(raw_edge[minr:maxr, minc:maxc])
#     #         ##plti(region_data[1], cmap='gray_r')
#     #         acc_rate = np.sum(np.abs(extract_label - region_data[1])) / (
#     #                     extract_label.shape[0] * extract_label.shape[1])
#     #         if acc_rate < 0.5:
#     #             rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red',
#     #                                      linewidth=2)
#     #             ax.add_patch(rect)
#     #
#     # ax.set_axis_off()
#     # plt.tight_layout()
#     # plt.show()
#
#
#
#     data = np.load('temp.npy')
#     plti(data)
#
#
#     #plti(data)
#     gray_scaleImage = find_section_connected_component(data, 60, 60,average_hu)
#     #plti(gray_scaleImage)
#     edges2 = filters.median(data, morphology.disk(1))
#     #plti(edges2)
#     edge = filters.sobel(edges2)
#     #plti(edge)
#
#     T = filters.threshold_isodata(edge)
#     edge = (edge < T) * 1.0
#     #plti(edge)
#     #plti(gray_scaleImage*edge)







# same_filename = "22 naive_AlexaFluor 647.tif"
# filename = "28 scAAV8-smCBA-mCherry B6_Alexa Fluor 647.tif"
# label_name = "large_high_pixel_labels"
# test_file = "test_paart.png"
# im = io.imread(filename)
# im = im.T
#
#
# test_data = im
#
# fig, ax = plt.subplots(figsize=(im.shape[1] / 100, im.shape[0] / 100))
# plt.imshow(im,cmap='gray')
# plt.show()
#
#
#
# sample_section = io.imread(same_filename)
# sample_section = sample_section.T
# #threshold_li = int(filters.threshold_li(test_data))
# plti(test_data)
#
#
# # average_width, average_hight, average_thresh,extract_label,average_hu,daisy = overall_label_information(label_name, same_filename)
# # canvas = np.load('xxx11.npy')
# # RESULT = overall_quantification(test_data,canvas,average_thresh,daisy)
# # boundary_test(average_hu)
# #overall_quantification(test_data,test_data, average_thresh,daisy)
# #io.imsave('Figure1.jpg',test_data)
#
# #
# # fig, ax = plt.subplots(figsize=(im.shape[0]/100, im.shape[1]/100))
# # ax.imshow(im)
#
# # plt.figure(figsize=(float(im.shape[1])/1440, float(im.shape[0])/1440), dpi=1440)
# # plt.savefig('myfig.png', dpi=1440)
#
# # ax.set_axis_off()
# # plt.tight_layout()
# # plt.savefig("ffff.png")
# # plt.show()
#
#
# group_labels, var,average_width = residual_groups(same_filename)
# final_coefficent, y_min_residual,final_curve_x, final_curve_y, function_curve_x, function_curve_y = residual_variance(group_labels)
#
#
#
#
#
# scan = IS()
# scan.__int__(average_width, final_coefficent, 0.1, y_min_residual)
# average_width, average_hight, average_thresh,extract_label,average_hu,daisy = overall_label_information(label_name, same_filename)
# feature_point = scan.scan(test_data,final_curve_x,final_curve_y, average_width, average_hight, average_thresh)
# section_list = mark_up(test_data,feature_point,round(average_width*0.8),round(average_hight* 0.8),average_hu, scale_coefficent=2)
# canvas = combine_section(section_list, test_data.shape[0],test_data.shape[1])
#
# RESULT = overall_quantification(test_data,canvas,average_thresh,daisy)
# print(RESULT)
# np.save("xxx11.npy",canvas)
# # im = im.T
# # print(im.shape)
# # labels = read_labels(label_file="Label_pairs")
# # cut_labels(labels,im)


#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class Cluster:
    '''
    
    '''
    def __init__(self, pixel_size=0.01, dilation_size=None):
        '''
        Constructor method for initializing the BBox class.

        Parameters:
        - self: The self reference.
        - pixel_size(float): The pixel size for an image.
        - dilation (int): The dilation size for each pixel. 
        '''
        self.pixel_size = pixel_size
        self.dilation_size = dilation_size
        self.new_data = None
        self.labels = None
        self.stats = None
        self.region_dict = {}
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0
        self.indices = 0
        self.object_height_buffer = 0.1
        

    def compute_regions(self, data):
        '''
        Function that clusters pixels from an image.
        Parameters:
        - self: The self reference.
        - data (Numpy Array): Array that 
        '''
        ## Normalize the data into integer buckets
        self.new_data = np.array([(int(x / self.pixel_size), int(y / self.pixel_size)) for x, y, z in data])
        self.min_x, self.min_y = np.min(self.new_data[:, 0]), np.min(self.new_data[:, 1])

        self.new_data[:, 0] -= self.min_x
        self.new_data[:, 1] -= self.min_y

        self.max_x, self.max_y = np.max(self.new_data[:, 0]), np.max(self.new_data[:, 1])

        ## Make an image and populate it.
        image = np.zeros((self.max_x + 1, self.max_y + 1), dtype=np.uint8)
        for x,y in self.new_data:
            image[x, y] = 255

        # fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        # axs[0].imshow(image)
        # axs[0].title.set_text('Initial Image')

        ## Dilate and erode image to get rid of gaps
        if self.dilation_size:
            kernel = np.ones((self.dilation_size, self.dilation_size), np.uint8)
            image = cv.dilate(image, kernel, iterations=1)
            image = cv.erode(image, kernel, iterations=1)

        # axs[1].imshow(image)
        # axs[1].title.set_text(f'After dilation and erosion ({self.dilation_size})')

        n_labels, self.labels, self.stats, centroids  = cv.connectedComponentsWithStats(image, 8, cv.CV_8U)
        # print(f'{n_labels} total regions')
    
        ## Remove components that are too big (tabletop) or too small (noise)
        largest_area_index = np.argmax(self.stats[:, 4])
        mask = np.logical_or(self.stats[:, 4] <= 10, self.stats[:, 4] == self.stats[largest_area_index, 4])

        ## Get the indices of the components that are too big or too small
        self.indices = np.where(mask)[0]
        # print(self.indices)
        # print(self.stats)
        # x_plot = []
        # y_plot = []
        # for x, y in centroids:
        #     x_plot.append(x)
        #     y_plot.append(y)

        # axs[2].imshow(self.labels)
        # axs[2].plot(x_plot, y_plot)
        # axs[2].title.set_text('Regions')
        # plt.show()


        # Initialize the regions dictionary
        self.region_dict = {i: {"points": []} for i in range(1, n_labels) if i not in self.indices}
        # self.regions = {i: [] for i in range(1, n_labels) if i not in self.indices}
        for X, Y, Z in data:
            region_id = self.check_point(X, Y)
            if region_id:
                self.region_dict[region_id]["points"].append([X, Y, Z])

        for id in self.region_dict:
            points_array = np.array(self.region_dict[id]["points"])
            x_centroid, y_centroid, _ = np.mean(points_array, axis=0)
            z_values = [point[2] for point in self.region_dict[id]["points"]]
            max_z_value = max(z_values)
        
            self.region_dict[id]["centroid"] = [round(x_centroid,3), round(y_centroid,3), round(abs(max_z_value) + self.object_height_buffer,3)]

        return self.region_dict


    def check_point(self, x, y):
        ## Normalize the point
        int_x = int(x / self.pixel_size)
        int_y = int(y / self.pixel_size)

        ## min_x, min_y = np.min(self.new_data[:, 0]), np.min(self.new_data[:, 1])
        int_x -= self.min_x
        int_y -= self.min_y
        if 0 <= int_x < self.labels.shape[0] and 0 <= int_y < self.labels.shape[1]:
            label = self.labels[int_x, int_y]
            if label not in self.indices:
                return label
            

if __name__ == '__main__':
    # data = load_data()
    cluster = Cluster(pixel_size=0.005, dilation_size=8)
  



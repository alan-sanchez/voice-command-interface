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
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.indices = None
        self.stats = None

    def fit(self, data):
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
        mask = np.logical_or(self.stats[:, 4] <= 30, self.stats[:, 4] == self.stats[largest_area_index, 4])

        ## Get the indices of the components that are too big or too small
        self.indices = np.where(mask)[0]
        # print(self.indices)
        # print(self.stats)

        # axs[2].imshow(self.labels)
        # axs[2].title.set_text('Regions')
        # plt.show()

        ## Initialize the regions dictionary
        self.regions = {i: [] for i in range(1, n_labels) if i not in self.indices}

        for X, Y, Z in data:
            label = self.check_point(X, Y, Z)
            if label:
                self.regions[label].append([X, Y, Z])

        return self.regions


    def check_point(self, x, y, z):
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


# def load_data():
#     data = []
#     with open('coordinates.txt', 'r') as f:
#         for line in f:
#             x, y = line.split()
#             data.append((float(x), float(y)))
    
#     return np.array(data)


if __name__ == '__main__':
    # data = load_data()
    cluster = Cluster(pixel_size=0.005, dilation_size=8)
  



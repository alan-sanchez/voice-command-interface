#!/usr/bin/env python3


import numpy as np

import cv2 as cv

import matplotlib.pyplot as plt


class Cluster:
    def __init__(self, pixel_size=0.01, dilation_size=None):
        self.pixel_size = pixel_size
        self.dilation_size = dilation_size
        self.new_data = None
        self.labels = None
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None

    def fit(self, data):
        # Normalize the data into integer buckets
        self.new_data = np.array([(int(x / self.pixel_size), int(y / self.pixel_size)) for x, y in data])
        self.min_x, self.min_y = np.min(self.new_data[:, 0]), np.min(self.new_data[:, 1])

        self.new_data[:, 0] -= self.min_x
        self.new_data[:, 1] -= self.min_y

        self.max_x, self.max_y = np.max(self.new_data[:, 0]), np.max(self.new_data[:, 1])

        # Make an image and populate it.
        image = np.zeros((self.max_x + 1, self.max_y + 1), dtype=np.uint8)
        for x,y in self.new_data:
            image[x, y] = 255

        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[0].imshow(image)
        axs[0].title.set_text('Initial Image')

        # Dilate and erode image to get rid of gaps
        if self.dilation_size:
            kernel = np.ones((self.dilation_size, self.dilation_size), np.uint8)
            image = cv.dilate(image, kernel, iterations=1)
            image = cv.erode(image, kernel, iterations=1)

        axs[1].imshow(image)
        axs[1].title.set_text(f'After dilation and erosion ({self.dilation_size})')

        n_labels, self.labels, stats, centroids  = cv.connectedComponentsWithStats(image, 8, cv.CV_8U)
        # print(f'{n_labels} total regions')
       
        # print(stats)

        #Remove components that are too big (tabletop) or too small (noise)
        print(f'{sum(np.logical_and(stats[:, 4] > 10, stats[:, 4] < 1000))} actual objects')

        axs[2].imshow(self.labels)
        axs[2].title.set_text('Regions')
        plt.show()

    def check_point(self, x, y):
        # Normalize the point
        int_x = int(x / self.pixel_size)
        int_y = int(y / self.pixel_size)
        
        # min_x, min_y = np.min(self.new_data[:, 0]), np.min(self.new_data[:, 1])
        int_x -= self.min_x
        int_y -= self.min_y
        print(int_x,int_y)
        # Check if the point is within the image bounds
        if 0 <= int_x < self.labels.shape[0] and 0 <= int_y < self.labels.shape[1]:
            label = self.labels[int_x, int_y]
            if label > 0:
                return True, label
            else:
                return False, None
        else:
            return False, None


def load_data():
    data = []
    with open('coordinates.txt', 'r') as f:
        for line in f:
            x, y = line.split()
            data.append((float(x), float(y)))
    
    return np.array(data)


if __name__ == '__main__':
    data = load_data()

    cluster = Cluster(pixel_size=0.005, dilation_size=8)
    cluster.fit(data)

    # Example usage of check_point method
    x, y = .55, .19  # Example coordinates
    is_within_label, label = cluster.check_point(x, y)
    if is_within_label:
        print(f'Point ({x}, {y}) is within label {label}')
    else:
        print(f'Point ({x}, {y}) is not within any label')
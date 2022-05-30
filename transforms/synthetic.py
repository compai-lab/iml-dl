import numpy as np
import cv2


class SyntheticSprites:
    """
    Adds sprite abnormalities to the image
    """
    def __init__(self, intensity_ranges=[(0, 0.25), (0.25, 0.75), (0.75, 1)],
                 center_coord_range=(16, 48), axes_length_range=(4, 2), angle_range=(30, 330)):
        """

        :param intensity_ranges: list of [tuples(mean, std)]
            mean and std for a normal distribution of intensities
        :param center_coord_range: tuple(min, max)
            min and max for an uniform distribution
        :param axes_length_range: tuple(mean, std)
            mean and std for a normal distribution of lesion size
        :param angle_range: tuple(min,max)
            min and max of the desired angle
        """
        super(SyntheticSprites, self).__init__()
        self.intensity_ranges = intensity_ranges
        self.center_coord_range = center_coord_range
        self.axes_length_range = axes_length_range
        self.angle_range = angle_range
        self.startAngle = 0
        self.endAngle = 360
        self.thickness = -1
        self.color_mask = 1

    def __call__(self, img):
        """
        :param img: 2D image
        :return: 2D image with abnormalities and 2D mask
        """
        new_axis = False
        if len(img.shape) > 2:
            new_axis = True
            img = img[0]
        mask = np.zeros(img.shape)
        for i in range(len(self.intensity_ranges)):
            center_coordinatesAr = np.random.randint(self.center_coord_range[0], self.center_coord_range[1], 2)
            center_coordinates = (center_coordinatesAr[0], center_coordinatesAr[1])

            axesLengthAr = np.abs(np.random.normal(self.axes_length_range[0], self.axes_length_range[1], 2))
            axesLength = (int(axesLengthAr[0]), int(axesLengthAr[1]))

            angle = np.random.randint(self.angle_range[0], self.angle_range[1])
            color = np.minimum(1, np.abs(np.random.uniform(self.intensity_ranges[i][0], self.intensity_ranges[i][1])))

            img = cv2.ellipse(img, center_coordinates, axesLength, angle, self.startAngle, self.endAngle, color,
                              self.thickness)

            mask = cv2.ellipse(mask, center_coordinates, axesLength, angle, self.startAngle, self.endAngle,
                               self.color_mask, self.thickness)
        img = img[np.newaxis, :] if new_axis else img
        mask = mask[np.newaxis, :] if new_axis else mask
        return img, mask


class SyntheticRect:
    def __init__(self, size_range=(5, 25)):
        super(SyntheticRect, self).__init__()
        self.size_range = size_range

    def __call__(self, x):
        img_size = x.shape[-1]
        width = np.random.randint(10, 30, (x.shape[0], x.shape[1]))
        height = np.random.randint(10, 30, (x.shape[0], x.shape[1]))
        start_x = np.random.randint(int(img_size / 8), img_size - width - 1, (x.shape[0], x.shape[1]))
        start_y = np.random.randint(int(img_size / 8), img_size - height - 1, (x.shape[0], x.shape[1]))
        intensity = np.random.uniform(0, 1, (x.shape[0], x.shape[1]))
        # # print(f'Synthetic Rectangle Generation: {width}- {height} + {intensity}')
        # #
        for b in range(len(x)):
            x[b, 0, start_x[b, 0]:start_x[b, 0] + width[b, 0], start_y[b, 0]:start_y[b, 0] + height[b, 0]] = intensity[
                b, 0]
        return x

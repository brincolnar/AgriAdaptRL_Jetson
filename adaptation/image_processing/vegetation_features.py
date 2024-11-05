from PIL import Image

import matplotlib.pyplot as plt
import cv2
import numpy as np

class VegetationIndices:
    def __init__(self, image):
        self.image = image
    
    def excess_green_index(self):
        red_channel = self.image[:,:,0]
        green_channel = self.image[:,:,1]
        blue_channel = self.image[:,:,2]
        
        green_excess_index = 2 * green_channel - red_channel - blue_channel
        # new_image = Image.fromarray(green_excess_index)
        # plt.title("Excess Greed Index Visualisation")
        # plt.imshow(new_image, cmpa="gray")
        # print("Excess green index: \n", green_excess_index)
        
        return green_excess_index
    
    def excess_red_index(self):
        red_channel = self.image[:,:,0]
        green_channel = self.image[:,:,1]
        
        red_excess_index = 1.4 * red_channel - green_channel
        # new_image = Image.fromarray(red_excess_index)
        # plt.title("Excess Red Index Visualisation")
        # plt.imshow(new_image)
        # print("Excess red index: \n", red_excess_index)
        
        return red_excess_index
        
    def colour_index_vegetation_extraction(self):
        red_channel = self.image[:,:,0]
        green_channel = self.image[:,:,1]
        blue_channel = self.image[:,:,2]

        cive = red_channel * 0.441 - green_channel * 0.811 + blue_channel * 0.385 + 18.78745
        new_image = Image.fromarray(cive)
        # plt.imshow(new_image)
        # plt.axis("off")
        # print("Colour index of Vegetation Extraction: \n", cive)

        return cive

    def excess_green_excess_red_index(self, excess_green, excess_red):
        diff = excess_green - excess_red
        _, binary_ExG_ExR = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)
        
        total_pixels = binary_ExG_ExR.size
        vegetation_pixels = np.count_nonzero(binary_ExG_ExR)
        vegetation_ratio = vegetation_pixels / total_pixels

        # print("ExG-ExR index Vegetation Ratio:", np.array("{:.3f}".format(vegetation_ratio)))
        
        # plt.title("ExG - ExR Binary")
        # plt.imshow(binary_ExG_ExR, cmap="gray")
        # print("Binary ExG-ExR index","\n", binary_ExG_ExR)

        return vegetation_ratio
        
    def visualization_CIVE_Otsu_threshold(self, cive_index):
        _, binary_img = cv2.threshold(cive_index.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        total_pixels = binary_img.size
        vegetation_pixels = np.count_nonzero(binary_img)
        vegetation_ratio = vegetation_pixels / total_pixels

        # print("CIVE Vegetation Ratio:", np.array("{:.3f}".format(vegetation_ratio)))

        # cv2.imshow("CIVE with Otsu Threshold", binary_img)
        # cv2.imwrite("CIVE_Otsu.png", binary_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return vegetation_ratio
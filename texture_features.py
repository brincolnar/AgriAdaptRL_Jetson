from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np

class TextureFeatures:
    def __init__(self, image):
        self.image = image

    def compute_glcm(self):
        gray = color.rgb2gray(self.image)
        image = img_as_ubyte(gray)
        d = [1]
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)
        max_value = inds.max()+1
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(inds, d, angles, levels=max_value, symmetric=True, normed=True)
        return glcm

    def contrast_feature(self, matrix_coocurrence):
        contrast = graycoprops(matrix_coocurrence, 'contrast')
        return contrast

    def dissimilarity_feature(self, matrix_coocurrence):
        dissimilarity = graycoprops(matrix_coocurrence, 'dissimilarity')
        return dissimilarity

    def homogeneity_feature(self, matrix_coocurrence):
        homogeneity = graycoprops(matrix_coocurrence, 'homogeneity')
        return homogeneity

    def energy_feature(self, matrix_coocurrence):
        energy = graycoprops(matrix_coocurrence, 'energy')
        return energy

    def correlation_feature(self, matrix_coocurrence):
        correlation = graycoprops(matrix_coocurrence, 'correlation')
        return correlation

    def asm_feature(self, matrix_coocurrence):
        asm = graycoprops(matrix_coocurrence, 'ASM')
        return asm

    def display(self, glcm):
        image = io.imread(self.image_path)
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(glcm[:, :, 0, 0], cmap=plt.cm.gray)
        plt.title('Grey-level co-occurence matrix')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

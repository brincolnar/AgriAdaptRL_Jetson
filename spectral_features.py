import cv2
import numpy as np

class SpectralFeatures:
   def __init__(self, image):
      self.image = image

   def compute_brightness(self):
      image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
      (Y,U,V) =  cv2.split(image.astype("float"))
      mean = "{:.3f}".format(np.mean(Y))
      std = "{:.3f}".format(np.std(Y))
      max_brightness = Y.max()
      min_brightness = Y.min()

      # print(f"mean brightness: {mean}")
      # print(f"standard deviation - brightness: {std}")
      # print(f"max brightness: {max_brightness}")
      # print(f"min brightness: {min_brightness}")

      return mean,std,max_brightness,min_brightness

   def compute_hue_histogram(self):
      image =  cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
      (H,S,V) =  cv2.split(image.astype("float"))
      hist = cv2.calcHist(image, [0], None,[20],[0,256])
      c_threshold = 0.01
      maximum = hist.max()
      feature_1 = np.sum([1 if hist[i]>=(c_threshold*maximum) else 0 for i in range(20)])
      max_2 = -1

      for i in range(20):
         if hist[i]==maximum:
            continue
         if hist[i]>max_2:
            max_2 = hist[i]
         feature_2 = maximum-max_2

      #  print(f"Number of bins: {feature_1}")
      #  print(f"Contrast of the hue histogram as the maximum arc length distance between any two significant bins: {feature_2[0]}")
      #  print(f"Standard deviation of the hue arc length of all the pixels in the image: {np.std(H)}")


      return feature_1,"{:.3f}".format(feature_2[0]),"{:.3f}".format(np.std(H))
   
   def compute_contrast(self):
      image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
      (Y,U,V) =  cv2.split(image.astype("float"))
      std = np.std(Y)
      maximum = Y.max()
      minimum = Y.min()
      if (maximum-minimum)<=0:
        return 0
      contrast = std*1.0/(maximum-minimum) 

      # print(f"Contrast: {contrast}")

      return "{:.3f}".format(contrast)
   
   def compute_saturation(self):
      image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
      (H,S,V) =  cv2.split(image.astype("float"))
      mean = np.mean(S)
      std = np.std(S)
      max_saturation = S.max()
      min_saturation = S.min()

      # print(f"mean saturation: {mean}")
      # print(f"standard deviation - saturation: {min_saturation}")
      # print(f"max saturation: {max_saturation}")
      # print(f"min saturation: {min_saturation}")

      return "{:.3f}".format(mean),"{:.3f}".format(std), max_saturation, min_saturation
   
   def compute_sift_feats(self):
      gray= cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
      sift = cv2.SIFT_create()
      kp = sift.detect(gray,None)
      no_kp = len(kp)

      # print("No of keypoints: ", no_kp)

      return no_kp
   
   def np_array_to_string(self, array):
      return ','.join(map(str, array))
   
   def flatten_array(self, input):
         return np.ravel(input)
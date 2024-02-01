import cv2  
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ImageTransformations:

    def __init__(self, image_path_or_object=None, RGB=True, read_image_from_file=True):
        """
        - put RGB=True to process the image as an RGB image (if it is) or otherwise put RGB=False to process the image as a Grayscale
        image (if is already in grayscale, it remains in grayscale, otherwise if it is in RGB scale it will be converted in Grayscale)
        """
        if read_image_from_file==True:
            if RGB == True:
                self.image = cv2.cvtColor(cv2.imread(image_path_or_object, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                self.RGB = True
            else:
                self.image = cv2.imread(image_path_or_object, cv2.IMREAD_GRAYSCALE)
                self.RGB = False
        else:
            if RGB == True:
                self.image = image_path_or_object
                self.RGB = True
            else:
                self.image = image_path_or_object
                self.RGB = False

    def set_image(self, image):
        self.image = image
        return image

    def show_image(self):
        if self.image is None:
            raise ValueError("Impossibile caricare l'immagine.")
        else:
            cv2.imshow('Immagine', self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def addition(self, value, value_G=None, value_B=None):
        all_parameters_filled=False
        if(value_G!=None and value_B!=None):
            all_parameters_filled=True
        if self.RGB == False:
             return ImageTransformations(cv2.add(self.image, value),self.RGB,False)
        else:
            if all_parameters_filled==False:
                return ImageTransformations(cv2.add(self.image,(value, value, value)),self.RGB,False)
            else:
                return ImageTransformations(cv2.add(self.image,(value, value_G, value_B)),self.RGB,False)
        
    def multiplication(self, value, value_G=None, value_B=None):
        all_parameters_filled=False
        if(value_G!=None and value_B!=None):
            all_parameters_filled=True
        if self.RGB == False:
            return ImageTransformations(cv2.multiply(self.image, value),self.RGB,False)
        else:
            if all_parameters_filled==False:
                return ImageTransformations(cv2.multiply(self.image,(value, value, value)),self.RGB,False)
            else:
                return ImageTransformations(cv2.multiply(self.image,(value, value_G, value_B)),self.RGB,False)
    
    def clamping(self, lower_parameters, upper_parameters):
        """
        - lower_parameters and upper_parameters are two lists, if use an image RGB specifing just two lists with one parameter
        will be used the same parameter for all the 3 channels, instead if you specify all the 3 numbers or the list you can 
        do a custom clamping giving more weight to a channel respect to another
        """
        if (self.RGB == False):
            if (len(lower_parameters)==1 and len(upper_parameters)==1):
                return ImageTransformations(np.clip(self.image, lower_parameters[0], upper_parameters[0]).astype(np.uint8),self.RGB,False)
            else: 
                raise ValueError("Insert a list of 1 element!") 
        else:
            if(len(lower_parameters)==1 and len(upper_parameters)==1):
                return ImageTransformations(np.clip(self.image,lower_parameters*3,upper_parameters*3).astype(np.uint8),self.RGB,False)
            elif(len(lower_parameters)==3 and len(upper_parameters)==3):
                return ImageTransformations(np.clip(self.image,lower_parameters,upper_parameters).astype(np.uint8),self.RGB,False)
            else:
                raise ValueError("Insert a list of 1 or 3 elements!") 
               
    def invert(self):
        return ImageTransformations((255 - self.image),self.RGB,False)
    
    def gamma_correction(self,gamma):
        return ImageTransformations(np.array(255*(self.image / 255) ** gamma, dtype = 'uint8').astype(np.uint8),self.RGB,False)

    def contrast_stretching(self,a,b):
        """
        - a represents the lower bound and b the upper bound on the x axe
        """
        c = int(self.image.min())
        d = int(self.image.max())
        return ImageTransformations(cv2.convertScaleAbs(cv2.subtract(self.image, c), alpha=((b - a)/(d - c)), beta=a),self.RGB,False)
    
    def histogram_equalization(self):
        if self.RGB==False:
           return ImageTransformations(cv2.equalizeHist(self.image),self.RGB,False)
        else:
           r, g, b = cv2.split(self.image)
           r_equalized = cv2.equalizeHist(r)
           g_equalized = cv2.equalizeHist(g)
           b_equalized = cv2.equalizeHist(b)
           return ImageTransformations(cv2.merge([r_equalized, g_equalized, b_equalized]),self.RGB,False)
        
    def clahe(self,clip_limit=2.0,grid_size=(8,8)):
        """
        - clip_limit is a float number: the more increase this number the more will be the equalization effect of the histogram
        - grid_size is a tuple with two numbers (ex. (8,8)) that represents a grid: the more are the raws and columns, the more the
        equalization will be calculated on bigger regions of the image 
        """
        if self.RGB==False:
            return ImageTransformations(cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(self.image),self.RGB,False)
        else:
           r, g, b = cv2.split(self.image)
           r_clahe_equalized = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(r)
           g_clahe_equalized = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(g)
           b_clahe_equalized = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(b)
           return ImageTransformations(cv2.merge([r_clahe_equalized, g_clahe_equalized, b_clahe_equalized]),self.RGB,False)
        

    def binary_treshold(self,treshold):
        if self.RGB==False:
            return ImageTransformations(cv2.threshold(self.image, treshold, 255, cv2.THRESH_BINARY)[1],self.RGB,False)
        else:
            r, g, b = cv2.split(self.image)
            _,r_binarized = cv2.threshold(r, treshold, 255, cv2.THRESH_BINARY)
            _,g_binarized = cv2.threshold(g, treshold, 255, cv2.THRESH_BINARY)
            _,b_binarized = cv2.threshold(b, treshold, 255, cv2.THRESH_BINARY)
            return ImageTransformations(cv2.merge([r_binarized, g_binarized, b_binarized]),self.RGB,False)
   
    def otsu_treshold(self):
        """- Otsu treshold is calculated automatically, you have not to set the treshold"""
        if self.RGB==False:
            return ImageTransformations(cv2.threshold(self.image, 0, 255, cv2.THRESH_OTSU)[1],self.RGB,False)
        else:
            r, g, b = cv2.split(self.image)
            _,r_otsu_binarized = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
            _,g_otsu_binarized = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
            _,b_otsu_binarized = cv2.threshold(b, 0, 255, cv2.THRESH_OTSU)
            return ImageTransformations(cv2.merge([r_otsu_binarized, g_otsu_binarized, b_otsu_binarized]),self.RGB,False)

    def adaptive_mean_treshold(self,subtract_to_mean=5,window_size=9):
        """ 
        - Attention! window_size must be an odd number (IT: numero dispari) greater than 1
        - The specified constant subtract_to_mean is subtracted from the calculated average to compensate the local variations in brightness or contrast.
        This helps stabilize the threshold in the presence of brightness variations.
        """
        if self.RGB==False:
            return ImageTransformations(cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window_size,subtract_to_mean),self.RGB,False)
        else:
            r, g, b = cv2.split(self.image)
            r_adaptive = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window_size,subtract_to_mean)
            g_adaptive = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window_size,subtract_to_mean)
            b_adaptive = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window_size,subtract_to_mean)
            return ImageTransformations(cv2.merge([r_adaptive, g_adaptive, b_adaptive]),self.RGB,False)

    def linear_blending(self,image_path_2,alpha=0.5):
        if self.RGB==True:    
            image2 = cv2.cvtColor(cv2.imread(image_path_2, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            image2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
        height, width = self.image.shape[:2]
        resized_image2= cv2.resize(image2,(width,height))
        return ImageTransformations(cv2.addWeighted(self.image, alpha, resized_image2, 1 - alpha, 0),self.RGB,False)
    
    def mean_filter(self, kernel_size=3):
        if kernel_size % 2 == 0:
            raise ValueError("The kernel size must be an odd number!")
        return ImageTransformations(cv2.blur(self.image, (kernel_size, kernel_size)),self.RGB,False)
    
    def median_filter(self, kernel_size=3):
        if kernel_size % 2 == 0:
            raise ValueError("The kernel size must be an odd number!")
        return ImageTransformations(cv2.medianBlur(self.image, kernel_size),self.RGB,False)
    
    def gaussian_filter(self, kernel_size=3, std_dev=1):
        if kernel_size % 2 == 0:
            raise ValueError("The kernel size must be an odd number!")
        return ImageTransformations(cv2.GaussianBlur(self.image, (kernel_size, kernel_size), std_dev),self.RGB,False)

    def crop(self, x, y, width, height):
        """
        Crops an image to the specified size.
        Parameters:
        - x, y: the coordinates of the starting point of the crop
        - width, height: the dimensions of the crop
        """
        return ImageTransformations(self.image[y:y+height, x:x+width],self.RGB,False)

    def padding(self,padding_size=2, padding_value=0):
        """
        Adds a padding to the image with the specified value.
        Parameters:
        - padding_size: the size of the padding to add on each side
        - padding_value: the padding value (default: 0)
        """
        return ImageTransformations(cv2.copyMakeBorder(self.image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=padding_value),self.RGB,False)

    def extend(self, extension_size=2):
        """
        Extends the image by duplicating pixels along each side.
        Parameters:
        - extension_size: the size of the extension to add on each side
        """
        return ImageTransformations(cv2.copyMakeBorder(self.image, extension_size, extension_size, extension_size, extension_size, cv2.BORDER_REPLICATE),self.RGB,False)

    def wrap(self, wrap_size):
        """
        Wraps the image using existing pixels along each side.
        Parameters:
        - wrap_size: the size of the wrap to add on each side
        """
        return ImageTransformations(cv2.copyMakeBorder(self.image, wrap_size, wrap_size, wrap_size, wrap_size, cv2.BORDER_WRAP),self.RGB,False)

    def get_shape(self):
        return self.image.shape   
    
    def cvt_rgb2gray(self):
        if self.image.ndim == 2: 
            raise ValueError("The image must be RGB!")
        else:  
            return ImageTransformations(cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY),not self.RGB,False)
        
    def cvt_gray2rgb(self):
        if self.image.ndim == 2: 
            return ImageTransformations(cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB),not self.RGB,False)
        else:  
            raise ValueError("The image must be Grayscale!")
    
    def is_RGB(self):
        return self.RGB
    
    def get_ndim(self):
        ndim= self.image.ndim
        return ndim
    
    def prewitt_filter_x(self):
        """
        Apply the Prewitt filter with respect to x axe to an image.
        Returns:
        - prewitt_x: image with the Prewitt filter applied along the x (horizontal) axis
        It is recommended to use the image in grayscale, so convert the image if necessary.
        """
        kernel_prewitt_x = np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]])/6.0
        return ImageTransformations(np.abs(cv2.filter2D(self.image, cv2.CV_64F, kernel_prewitt_x)).astype(np.uint8),self.RGB,False)
    
    def prewitt_filter_y(self):
        """
        Apply the Prewitt filter with respect to y axe to an image.
        Returns:
        - prewitt_y: image with the Prewitt filter applied along the y axis (vertical).
        It is recommended to use the image in grayscale, so convert the image if necessary.
        """
        kernel_prewitt_y = np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]])/6.0
        return ImageTransformations(np.abs(cv2.filter2D(self.image, cv2.CV_64F, kernel_prewitt_y)).astype(np.uint8),self.RGB,False)

    def prewitt_filter(self):
        """
        Combine the prewitt x and y effect in one image.
        It is recommended to use the image in grayscale, so convert the image if necessary.
        """
        kernel_prewitt_x = np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]])/6.0    
        kernel_prewitt_y = np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]])/6.0

        # Combina i risultati per ottenere il filtro Prewitt totale
        prewitt_combined = np.hypot(np.abs(cv2.filter2D(self.image, cv2.CV_64F, kernel_prewitt_x)), np.abs(cv2.filter2D(self.image, cv2.CV_64F, kernel_prewitt_y))).astype(np.uint8)
        return ImageTransformations(prewitt_combined,self.RGB,False)

    def sobel_filter_x(self):
        """
        Apply the Sobel filter with respect to x axe to an image.
        It is recommended to use the image in grayscale, so convert the image if necessary.
        """
        return ImageTransformations(np.abs(cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)).astype(np.uint8),self.RGB,False)
    
    def sobel_filter_y(self):
        """
        Apply the Sobel filter with respect to y axe to an image.
        It is recommended to use the image in grayscale, so convert the image if necessary.
        """
        return ImageTransformations(np.abs(cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)).astype(np.uint8),self.RGB,False)
    
    def sobel_filter(self):
        """
        Combine the Sobel x and y effect in one image.
        It is recommended to use the image in grayscale, so convert the image if necessary.
        """
        sobel_combined = np.hypot(np.abs(cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)), np.abs(cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3))).astype(np.uint8)
        return ImageTransformations(sobel_combined,self.RGB,False)

    def laplacian_filter(self):
        """It is recommended to use the image in grayscale, so convert the image if necessary"""
        laplacian = cv2.Laplacian(self.image, cv2.CV_64F)
        return ImageTransformations(np.abs(laplacian).astype(np.uint8),self.RGB,False)
    
    def erode(self,kernel_size=3):
        """Take care that the image was binarized (tresholded) (it works also with RGB images)"""
        return ImageTransformations(cv2.erode(self.image, np.ones((kernel_size, kernel_size), np.uint8), iterations=1),self.RGB,False)

    def dilate(self,kernel_size=3):
        """Take care that the image was binarized (tresholded) (it works also with RGB images)"""
        return ImageTransformations(cv2.dilate(self.image, np.ones((kernel_size, kernel_size), np.uint8), iterations=1),self.RGB,False)

    def canny(self,lower_treshold, upper_treshold):
        """Take care that the image was binarized (tresholded) (it works also with RGB images)"""
        return ImageTransformations(cv2.Canny(self.image, lower_treshold, upper_treshold),self.RGB,False)


    def clahe_v2(self, clip_limit=2.0,grid_size=(8,8)):
        b_channel, g_channel, r_channel = cv2.split(self.image)

        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clip_limit, grid_size)
        clahe_b_channel = clahe.apply(b_channel)
        clahe_g_channel = clahe.apply(g_channel)
        clahe_r_channel = clahe.apply(r_channel)

        # Merge the channels back to form the CLAHE-enhanced BGR image
        self.image = cv2.merge([clahe_b_channel, clahe_g_channel, clahe_r_channel])
        return self.image




#Static methods
def show_images_with_histograms(input_image,modified_image):
    """Plot the two images with the relative histograms"""
   
    original_hist = cv2.calcHist([input_image], [0], None, [256], [0, 256])
    modified_hist = cv2.calcHist([modified_image], [0], None, [256], [0, 256])

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.1, 1])

    ax1 = plt.subplot(gs[0, 0])
    if input_image.ndim == 2: 
        ax1.imshow(input_image, cmap='gray')
    else:  
        ax1.imshow(input_image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(original_hist, color='green')
    ax2.set_title('Original Histogram')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')

    plt.subplot(gs[1, :]).axis('off')

    ax3 = plt.subplot(gs[2, 0])
    if modified_image.ndim == 2: 
        ax3.imshow(modified_image, cmap='gray')
    else:  
        ax3.imshow(modified_image)
    ax3.axis('off')

    ax4 = plt.subplot(gs[2, 1])
    ax4.plot(modified_hist, color='red')
    ax4.set_title('Modified Histogram')
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Frequency')

    plt.show()

def shot_two_images(input_image,image2, space=0.05):
    """Plot the two images one beside the other"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    axs[0].imshow(cv2.cvtColor(cv2.imread(input_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[0].axis('off')

    plt.subplots_adjust(wspace=space)

    axs[1].imshow(image2)
    axs[1].set_title('Modified')
    axs[1].axis('off')

    plt.show()











#TRYING

#When initialize the ImageTransormations object use the image path as first parameter, and indicate True or False if the image is RGB (True) or Grayscale (False)
#Don't touch other parameter of the constructor    
    
# frame= ImageTransformations('test.png',True)
# transform=frame.clahe()
# show_images_with_histograms(frame.image,transform.image)

#shot_two_images('input_image_RGB.jpg',frame.addition(50).clamping([30],[140]).image)


    
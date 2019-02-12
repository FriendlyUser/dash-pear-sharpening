#!/usr/bin/env python3
# David Li
# V00818631 
# ECE 471 --- Computer Vision
# Assignment 1
# This program should work with an kernel size

# To run: python3 solution.py --image <path to your image> 
#        python3 solution.py (uses default path = "pear.png")       

#Using "as" nicknames a library so you don't have to use the full name
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse as ap

#----------------CUSTOM FUNCTIONS -----------------------------------

# Simple function to arhibarily generate gaussian kernel
def gkern(l=9, sig=5.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    # Add Error handling for odd numbers
    if l % 2 == 0:
        # force length to be an odd number
        l = l + 1

    # Divide input length by 2 (floor) and plus 1
    # For input 9, gives np.arange(-5,-5) not inclusive
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    # Typical meshgrid from (-l/2 + 1, l/2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    # Compute gaussian kernel
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    # Divide by kernel sum, to ensure kernel adds to 1.
    return kernel / np.sum(kernel)

# Reference
# http://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/
def show_image_side_by_side(image_orig,image_sharp,image_title='Original Vs Sharpened Image'):
    """
    Useful function for comparing input images and output images
        Input:  im_orig original image
                im_sharp sharpened image
        Output: Grayscale images with the original and sharpened image side by side
    """
    image_original  = cv2.imread(image_orig)
    image_sharpened = cv2.imread(image_sharp)

    image_grey = cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY)
    # Make the grey scale image have three channels
    grey_3_channel = cv2.cvtColor(image_grey, cv2.COLOR_GRAY2BGR)

    numpy_vertical = np.vstack((image_original, grey_3_channel))
    numpy_horizontal = np.hstack((image_original, grey_3_channel))

    numpy_horizontal_concat = np.concatenate((grey_3_channel,image_sharpened), axis=1)

    cv2.imshow(image_title, numpy_horizontal_concat)
    cv2.waitKey(0)

def image_kernel_test(img,ksize):
    """
    Test function that computes a LoG, and convolves it using cv2.filter2D, only 
    run for debugging
    Input:  img = input HxWx1 grayscale image
            ksize = size of the kernel (default=use 9x9 kernel)
    """
    sigma = 5
    impulse = np.zeros((ksize,ksize))

    # Create impulse by setting center of zero matrix as 2
    impulse[ksize // 2 + 1, ksize // 2 + 1] = 2
    kernel = gkern(ksize,sigma) 
    # Compute approximate Laplacian of Gaussian (as stated in lecture notes 2)
    kernel = impulse - kernel
    test_img = cv2.filter2D(img,-1,kernel)
    show_image(test_img,'Testing Kernel Image')
    save_image(test_img,'sharpened_test.png')

#----------------Basic Image Handling (15 marks)---------------------
#5 marks: Read an image and return a grayscale image
#Input: image_path = string path to the image
#Output: HxWx1 numpy array containing the image
def read_image(image_path):
    # Read image as grayscale
    # Since I have extra time I wonder if I can revert back to basics and just read it without using cv2.imread, grayscale mode
    img = cv2.imread(image_path,0)
    return img

#5 marks: Save an image and return True is successful, False if not
#Input: image_to_save = 8 unsigned integer numpy array (image)
#       image_path = string path to save the image
#Output: True if image saved successfully, False otherwise
def save_image(image_to_save, image_path):
    # Add error handling
    try:
      cv2.imwrite(image_path,image_to_save)
    except Exception as e:
      print(e)
    

#5 marks: Display image (you can use either cv2 or matplotlib
#Input: image_to_show = image to display
def show_image(image_to_show,image_title="Pear Image",show_time=2000):
    # Add error handling
    cv2.imshow(image_title,image_to_show)
    # pauses for 2 seconds before fetching next image
    cv2.waitKey(show_time)
    cv2.destroyAllWindows()

#----------------Image Processing (35 marks)-------------------------
"""
Input: img = input HxWx1 grayscale image
       ksize = size of the kernel (default=use 11x11 kernel)
       padding = what kind of padding around the image do we want to use
Output: return unsigned 8b integer image
"""

def imgfilter2d(image,ksize=11,padding=cv2.BORDER_REFLECT):
    #5 Mark: Create sharpen filter with a 9x9 Gaussian kernel with sigma 5, and unit
    #        impulse of 2 

    sigma = 5
    impulse = np.zeros((ksize,ksize))

    # Create impulse by setting center of zero matrix as 2
    impulse[ksize // 2 + 1, ksize // 2 + 1] = 2
    kernel = gkern(ksize,sigma) 
    # Compute approximate Laplacian of Gaussian (as stated in lecture notes 2)
    kernel = impulse - kernel

    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (imageH, imageW) = image.shape[:2]
    (kernelH, kernelW) = kernel.shape[:2]
    result = np.zeros((imageH, imageW), dtype="float32")

    #5 Marks: Apply border padding to the image
    pad = (kernelH - 1) // 2
    image_padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, padding)

    # 20 marks: perform the convolution
    # 5 marks : sliding window that applies the convolution
    # 5 marks : get the values in the neighbourhood
    # 5 marks : perform the convolution
    # 5 marks : save the result
    # (BONUS 10 marks): your solution can handle other filter sizes

    # Standard convolution function taken from https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
    for y in np.arange(pad, imageH + pad):
        for x in np.arange(pad, imageW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
            roi = image_padded[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
            k = (roi * kernel).sum()

			# store the convolved value in the output (x,y)-
			# coordinate of the output image
            result[y - pad, x - pad] = k

    #3 Marks: clip the result so that the values are in the range (0,255) and save as unsigned 8 bit integer

    result_clipped = (result.clip(min=0,max=255)).astype("uint8")

    return result_clipped 

def str2bool(v):
    """
    Parsing for boolean in python
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # This library handles argument parsing.  You don't need to worry about this for this assignment.
    parser = ap.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path to image", default="pear.png") 

    # Add debug mode
    parser.add_argument("-d","--debug", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Run Debug Functions")
    args = parser.parse_args()
    image = read_image(args.image)
    print('=============Showing the Original Image========================')
    show_image(image)
    if args.debug:
        print('=============Testing Standard Kernel Size=============')
        image_kernel_test(image,9)
        show_image_side_by_side('pear.png','sharpened_test.png')
        # Testing bigger kernel size
        print('=============Testing Larger Kernel Size=============')
        image_kernel_test(image,101)
        show_image_side_by_side('pear.png','sharpened_test.png')
    else: 
        print('=============Calculating approx LoG And Convolution=============')
        res = imgfilter2d(image,9)
        print('=============Saving Image =============')
        show_image(res)
        save_image(res, "sharpened.png")
        print('============Showing Comparsion===========')
        show_image_side_by_side('pear.png','sharpened.png')
        print('============Script Complete===========')

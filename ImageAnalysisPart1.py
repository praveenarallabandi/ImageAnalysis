# imports
import os
import math
import numpy as np  
from numpy import asarray 
import matplotlib.pyplot as plt
import time
start_time = time.time()

# resolution for images
ROWS = 768    
COLS =  568
TotalPixels = ROWS * COLS
imageClasses = {}
# Process files in directory as a batch
def process_batch(path):
    # basepath = ('./Cancerouscellsmears2')
    with os.scandir(path) as entries:
        groupImageClass(entries)
    return basepath

def groupImageClass(entries):
    columnar, parabasal, intermediate, superficial, mild, moderate, severe = [],[], [], [], [], [], []

    for entry in entries:
        if entry.is_file():
            if entry.name.find('cyl') != -1:
                columnar.append(entry)
            
            if entry.name.find('inter') != -1:
                intermediate.append(entry)
            
            if entry.name.find('para') != -1:
                parabasal.append(entry)
            
            if entry.name.find('super') != -1:
                superficial.append(entry)

            if entry.name.find('let') != -1:
                mild.append(entry)

            if entry.name.find('mod') != -1:
                moderate.append(entry)

            if entry.name.find('svar') != -1:
                severe.append(entry)
        
    imageClasses['columnar'] = columnar
    imageClasses['parabasal'] = parabasal
    imageClasses['intermediate'] = intermediate
    imageClasses['superficial'] = superficial
    imageClasses['mild'] = mild
    imageClasses['moderate'] = moderate
    imageClasses['severe'] = severe

    # print('Keys - {}'.format(imageClasses.keys()))
    # print('Values - {}'.format(imageClasses.values()))
    for imageClass in imageClasses:
        # print('Processing Image - {}'.format(imageClasses[imageClass]))
        for image in imageClasses[imageClass]:
            print('Processing Image - {}'.format(image.name))
            process_image(image)
    print("--- %s seconds ---" % (time.time() - start_time))

# Process the input image
def process_image(entry):
    # Given images is 1D array
    # origImage = np.fromfile(entry, dtype = np.uint8, count = TotalPixels)
    origImage = plt.imread('./Cancerouscellsmears2/' + entry.name)
    print("--------------------ORIGINAL IMAGE--------------------")
    print("Size of the image array: ", origImage.size)
    print('Type of the image : ' , type(origImage)) 
    print('Shape of the image : {}'.format(origImage.shape)) 
    print('Image Height {}'.format(origImage.shape[0])) 
    print('Dimension of Image {}'.format(origImage.ndim))
    pltImage(origImage, 'Original Image')

    # Conversion from 1D to 2D array - All Gray scale images are in 2D array
    """ origImage.shape = (origImage.size // COLS, COLS)
    print("--------------------2D--------------------")
    print("Size of the image array: ", origImage.size)
    print('Shape of the image : {}'.format(origImage.shape)) 
    print('Image Height {}'.format(origImage.shape[0])) 
    print('Image Width {}'.format(origImage.shape[1])) 
    print('Dimension of Image {}'.format(origImage.ndim))
    pltImage(origImage, '2D') """
    
    # Conversion from 2D to 3D RGB
    """ orig3DImage = gray2rgb(origImage)
    print("--------------------3D--------------------")
    print("Size of the image array: ", orig3DImage.size)
    print('Shape of the image : {}'.format(orig3DImage.shape)) 
    print('Image Hight {}'.format(orig3DImage.shape[0])) 
    print('Image Width {}'.format(orig3DImage.shape[1])) 
    print('Dimension of Image {}'.format(orig3DImage.ndim))
    print('Maximum RGB value in this image {}'.format(orig3DImage.max())) 
    print('Minimum RGB value in this image {}'.format(orig3DImage.min())) """

    # Converting color images to selected single color spectrum
    convertToSingleColorSpectrum(origImage, 'R')
    convertToSingleColorSpectrum(origImage, 'G')
    convertToSingleColorSpectrum(origImage, 'B')

    # Noise addition functions that will allow to corrupt each image with Gaussian & SP
    print('--------------------NOISE--------------------')
    corruptImage('gaussian', origImage)
    corruptImage('sp', origImage)
    
    # Histogram calculation for each individual image
    print('--------------------HISTOGRAM--------------------')
    calc_histogram(origImage)

    # Selected image quantization technique for user-specified levels
    print('--------------------IMAGE QUANTIZATION--------------------')
    image_quantization(origImage, 0.5)

    # Linear filter with user-specified mask size and pixel weights
    print('--------------------FILTERING OPERATIONS--------------------')
    linearFilterGaussian(origImage, 3, 1.5)

    final(entry)
    
def calc_histogram(image):
    vals = image.mean(axis=2).flatten()
    hist, bins = np.histogram(vals, density=True)
    """ print('Hist Counts {}'.format(hist)) 
    print('Bins {}'.format(bins)) """
    print('Histogram Sum {}'.format(hist.sum())) 
    print('Result {}'.format(np.sum(hist * np.diff(bins)))) 
    eqHistogram = equalize_histogram(image, bins)
    print('Equalize Histograms {}'.format(eqHistogram)) 

def equalize_histogram(a, bins):
    # https://gist.github.com/TimSC/6f429dfacf523f5c9a58c3b629f0540e
	a = np.array(a)
	hist, bins2 = np.histogram(a, bins=bins)
	#Compute CDF from histogram
	cdf = np.cumsum(hist, dtype=np.float64)
	cdf = np.hstack(([0], cdf))
	cdf = cdf / cdf[-1]
	#Do equalization
	binnum = np.digitize(a, bins, True)-1
	neg = np.where(binnum < 0)
	binnum[neg] = 0
	aeq = cdf[binnum] * bins[-1]
	return aeq

def image_quantization(image, level):
    # https://stackoverflow.com/questions/38152081/how-do-you-quantize-a-simple-input-using-python - TODO
    result =  level * np.round(image/level) 
    print('Result {}'.format(result))

def convertToSingleColorSpectrum(orig3DImage, colorSpectrum):
    plt.ylabel('Height {}'.format(orig3DImage.shape[0])) 
    plt.xlabel('Width {}'.format(orig3DImage.shape[1])) 
    if(colorSpectrum == 'R') :
        print('Value of only R channel {}'.format(orig3DImage[10, 10, 0]))
        plt.title('R channel') 
        plt.imshow(orig3DImage[ : , : , 0])
        
    if(colorSpectrum == 'G') :
        print('Value of only G channel {}'.format(orig3DImage[10, 10, 1]))
        plt.title('G channel') 
        plt.imshow(orig3DImage[ : , : , 1])

    if(colorSpectrum == 'B') :
        print('Value of only B channel {}'.format(orig3DImage[10, 10, 2]))
        plt.title('B channel') 
        plt.imshow(orig3DImage[ : , : , 2])

    # plt.show() # UNCOMMENT THIS - TODO

def pltImage(image, title):
    plt.ylabel('Height {}'.format(image.shape[0])) 
    plt.xlabel('Width {}'.format(image.shape[1])) 
    plt.title(title) 
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def gray2rgb(image):
    """ width, height = image.shape
    out = np.empty((width, height, 3), dtype=np.uint8)
    out[:, :, 0] = image
    out[:, :, 1] = image
    out[:, :, 2] = image """
    # https://stackoverflow.com/questions/59219210/extend-a-greyscale-image-to-fit-a-rgb-image
    out = np.dstack((image, np.zeros_like(image) + 255, np.zeros_like(image) + 255)) 
    # out = np.dstack((image, image, image))
    return out

def rgb2gray(img):
    return np.dot(img[...,:3], [0.299, 0.587,0.114]).astype(np.uint8)
    # return np.dot(img, [0.2126, 0.7152, 0.0722])

def corruptImage(noise_typ, image):
    row,col,ch= image.shape
    if noise_typ == "gaussian":
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        print('>>>>>>>>>> Gaussian >>>>>>>>>>') 
        print(format(noisy)) 
        
    elif noise_typ == "sp":
        s_vs_p = 0.5
        amount = 0.004
        sp = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        # out[coords] = 1
        sp[tuple(coords)]
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        sp[coords] = 0
        print('>>>>>>>>>> Salt & Pepper >>>>>>>>>>') 
        print(format(sp)) 
        # return out

def linearFilterGaussian(image, size=5, sigma=1.):
    # converti to 2D gray image first
    gray = rgb2gray(image)
    print("--------------------2D - GRAY--------------------")
    print("Size of the image array: ", gray.size)
    print('Shape of the image : {}'.format(gray.shape)) 
    print('Image Height {}'.format(gray.shape[0])) 
    print('Image Width {}'.format(gray.shape[1])) 
    print('Dimension of Image {}'.format(gray.ndim))

    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    # https://stackoverflow.com/questions/47369579/how-to-get-the-gaussian-filter
    # https://github.com/joeiddon/rpi_vision/blob/master/test.py
    # https://www.google.com/search?q=apply+gaussian+filter+to+image+%2B+numoy&oq=apply+gaussian+filter+to+image+%2B+numoy&aqs=chrome..69i57j0i333.12931j0j1&sourceid=chrome&ie=UTF-8
    # https://stackoverflow.com/questions/29920114/how-to-gauss-filter-blur-a-floating-point-numpy-array
    kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    result = kernel / np.sum(kernel)
    
    a = np.apply_along_axis(lambda x: np.convolve(x, result.flatten(), mode='same'), 0, gray)
    a = np.apply_along_axis(lambda x: np.convolve(x, result.flatten(), mode='same'), 1, gray)
    print("linearFilterGaussian: ", a)
    pltImage(a, 'Linear Filter')
    

def final(entry):
    print("--- %s seconds ---" % (time.time() - start_time))
    entry.close()
    """ result2DGrayImg = rgb2gray(origImage)
    print('-----------GRAY SCALE---------------')
    print("Size of the image array: ", result2DGrayImg.size)
    print('Type of the image : ' , type(result2DGrayImg)) 
    print('Shape of the image : {}'.format(result2DGrayImg.shape)) 
    print('Image Hight {}'.format(result2DGrayImg.shape[0])) 
    print('Dimension of Image {}'.format(result2DGrayImg.ndim))
    plt.imsave('Cancerouscellsmears2/RAW/Test.png', result2DGrayImg, cmap='gray')
    entry.close()
    plt.imsave('Cancerouscellsmears2/RAW/' + entry.name + '.png', rgb2gray(origImage))
    print("... File successfully saved") """


print('----------IMAGE ANALYSIS-------------------')
path = input('Enter images relative path: ')
if(path == '') :
    path = './Cancerouscellsmears2'
basepath = process_batch(path)





       

        
       
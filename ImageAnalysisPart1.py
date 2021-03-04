# imports
import os
import numpy as np  
import matplotlib.pyplot as plt

# resolution for images
ROWS = 768    
COLS =  568
TotalPixels = ROWS * COLS

# Process files in directory as a batch
def process_batch(path):
    # basepath = ('./Cancerouscellsmears2')
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                # print('Processing Image - ' + )
                print('Processing Image - {}'.format(entry.name))
                process_image(entry)
    return basepath

# Process the input image
def process_image(entry):
    # Given images is 1D array
    origImage = np.fromfile(entry, dtype = np.uint8, count = TotalPixels)
    print("--------------------1D--------------------")
    print("Size of the image array: ", origImage.size)
    print('Type of the image : ' , type(origImage)) 
    print('Shape of the image : {}'.format(origImage.shape)) 
    print('Image Height {}'.format(origImage.shape[0])) 
    print('Dimension of Image {}'.format(origImage.ndim))

    # Conversion from 1D to 2D array - All Gray scale images are in 2D array
    origImage.shape = (origImage.size // COLS, COLS)
    print("--------------------2D--------------------")
    print("Size of the image array: ", origImage.size)
    print('Shape of the image : {}'.format(origImage.shape)) 
    print('Image Height {}'.format(origImage.shape[0])) 
    print('Image Width {}'.format(origImage.shape[1])) 
    print('Dimension of Image {}'.format(origImage.ndim))

    # Conversion from 2D to 3D RGB
    orig3DImage = gray2rgb(origImage)
    print("--------------------3D--------------------")
    print("Size of the image array: ", orig3DImage.size)
    print('Shape of the image : {}'.format(orig3DImage.shape)) 
    print('Image Hight {}'.format(orig3DImage.shape[0])) 
    print('Image Width {}'.format(orig3DImage.shape[1])) 
    print('Dimension of Image {}'.format(orig3DImage.ndim))

    # Converting color images to selected single color spectrum
    convertToSingleColorSpectrum(orig3DImage, 'R')
    convertToSingleColorSpectrum(orig3DImage, 'G')
    convertToSingleColorSpectrum(orig3DImage, 'B')

    # Noise addition functions that will allow to corrupt each image with Gaussian & SP
    print('--------------------NOISE--------------------')
    corruptImage('gaussian', origImage)
    corruptImage('sp', origImage)
    
    # Histogram calculation for each individual image
    print('--------------------HISTOGRAM--------------------')
    calc_histogram(orig3DImage)

    # Selected image quantization technique for user-specified levels
    print('--------------------IMAGE QUANTIZATION--------------------')
    image_quantization(orig3DImage, 150)
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
        print('Value of only G channel {}'.format(orig3DImage[1, 1, 1]))
        plt.title('G channel') 
        plt.imshow(orig3DImage[ : , : , 1])

    if(colorSpectrum == 'B') :
        print('Value of only B channel {}'.format(orig3DImage[1, 1, 2]))
        plt.title('B channel') 
        plt.imshow(orig3DImage[ : , : , 2])

    # plt.show() # UNCOMMENT THIS - TODO

def gray2rgb(image):
    """ width, height = image.shape
    out = np.empty((width, height, 3), dtype=np.uint8)
    out[:, :, 0] = image
    out[:, :, 1] = image
    out[:, :, 2] = image """
    # https://stackoverflow.com/questions/59219210/extend-a-greyscale-image-to-fit-a-rgb-image
    out = np.dstack((image, np.zeros_like(image) + 255, np.zeros_like(image) + 255)) 
    return out

def corruptImage(noise_typ, image):
   if noise_typ == "gaussian":
      row,col= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      print('>>>>>>>>>> Gaussian >>>>>>>>>>') 
      print(format(noisy)) 
      
   elif noise_typ == "sp":
      row,col = image.shape
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

def final(entry):
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

""" def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]) """

print('----------IMAGE ANALYSIS-------------------')
path = input('Enter images relative path: ')
if(path == '') :
    path = './Cancerouscellsmears2'
basepath = process_batch(path)





       

        
       
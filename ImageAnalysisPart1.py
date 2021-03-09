# imports
import os
import math
import numpy as np  
from numpy import asarray 
import matplotlib.pyplot as plt
import time
import csv

class InputFromCsv: 
    def __init__(self, path, noiseType, NoiseStrength, NoiseGMeanL, NoiseGSD, SingleColorSpectum, ImageQuantLevel): 
        self.path = path
        self.noiseType = noiseType
        self.NoiseStrength  = NoiseStrength
        self.NoiseGMeanL = NoiseGMeanL
        self.NoiseGSD = NoiseGSD
        self.SingleColorSpectum = SingleColorSpectum
        self.ImageQuantLevel = ImageQuantLevel

# This function returns an object of Test 
def getInput(path, noiseType, NoiseStrength, NoiseGMeanL, NoiseGSD, SingleColorSpectum, ImageQuantLevel): 
    return InputFromCsv(path, noiseType, NoiseStrength, NoiseGMeanL, NoiseGSD, SingleColorSpectum, ImageQuantLevel)

start_time = time.time()

# resolution for images
ROWS = 768    
COLS =  568
TotalPixels = ROWS * COLS
imageClasses = {}
imageClassesProcessTime = {}
temp = {}
imageNoisyPt = []
imageHistogramPt = []
imageSingleSpectrumPt = []
imageQuantizationPt = []
imageLinearFilterPt = []
imageMedianFilterPt = []

columnar = []
parabasal = []
intermediate = []
superficial = []
mild = []
severe = []


# Process files in directory as a batch
def process_batch(path, input):
    # basepath = ('./Cancerouscellsmears2')
    with os.scandir(path) as entries:
        groupImageClass(entries, input)

def groupImageClass(entries, input):
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

    for imageClass in imageClasses:
        for image in imageClasses[imageClass]:
            print('Processing Image - {}'.format(image.name))
            print('Processing Image Class - {}'.format(imageClass))
            print('Processing Input - {0}, {1}, {2}, {3}, {4}, {5}, {6}'.format(input.path, input.noiseType, input.NoiseStrength, input.NoiseGMeanL, input.NoiseGSD, input.SingleColorSpectum, input.ImageQuantLevel))
            process_image(image, imageClass, input)
        imageClassesProcessTime[imageClass] = (time.time() - start_time) % 60

    perf_metrics()

def perf_metrics():
    """ print ('imageHistogramPt {0}'.format(imageHistogramPt))
    print ('imageHistogramPt lenth {0}'.format(len(imageHistogramPt)))
    t = sum(imageHistogramPt)
    print ('imageHistogramPt Sum {0}'.format(t))
    a = t % len(imageHistogramPt)
    b = t / len(imageHistogramPt)
    print ('imageHistogramPt Avg  % {0}'.format(t))
    print ('imageHistogramPt Avg  / {0}'.format(b)) """
    print('********************************************************************')
    print('\t\t PERFORMANCE METRICS ')
    print('********************************************************************')
    print('--------------------------------------------------------------------')
    print('Total Processig time {}'.format(time.time() - start_time))
    print('--------------------------------------------------------------------')
    print('--------------------------------------------------------------------')
    print('Procedure \t Total Execution Time \t Average Time Per Image')
    print('--------------------------------------------------------------------')
    ans = sum(imageNoisyPt)
    avg = ans / len(imageNoisyPt)
    print('{0} \t\t {1} \t {2}'.format('Noise', ans, avg))
    ans = sum(imageHistogramPt)
    avg = ans / len(imageHistogramPt)
    print('{0} \t {1} \t {2}'.format('Histogram', ans, avg))
    ans = sum(imageSingleSpectrumPt)
    avg = ans / len(imageSingleSpectrumPt)
    print('{0}  {1} \t {2}'.format('Single Spectrum', ans, avg))
    ans = sum(imageQuantizationPt)
    avg = ans / len(imageQuantizationPt)
    print('{0} \t {1} \t {2}'.format('Quantization', ans, avg))
    ans = sum(imageLinearFilterPt)
    avg = ans / len(imageLinearFilterPt)
    print('{0} \t {1} \t {2}'.format('Linear Filter', ans, avg))
    ans = sum(imageMedianFilterPt)
    avg = ans / len(imageMedianFilterPt)
    print('{0} \t {1} \t {2}'.format('Median Filter', ans, avg))
    print('********************************************************************')
    for classTime in imageClassesProcessTime:
        print('Processig time for {0} - {1} seconds'.format(classTime, imageClassesProcessTime[classTime]))
    print('******************************* END *************************************')

# Process the input image
def process_image(entry, imageClass, input):
    # Given images is 1D array
    # origImage = np.fromfile(entry, dtype = np.uint8, count = TotalPixels)
    origImage = plt.imread(input.path + '/' + entry.name)
    print("--------------------ORIGINAL IMAGE--------------------")
    print("Size of the image array: ", origImage.size)
    print('Type of the image : ' , type(origImage)) 
    print('Shape of the image : {}'.format(origImage.shape)) 
    print('Image Height {}'.format(origImage.shape[0])) 
    print('Image Width {}'.format(origImage.shape[1]))
    print('Dimension of Image {}'.format(origImage.ndim))
    pltImage(origImage, 'Original Image')

    # Noise addition functions that will allow to corrupt each image with Gaussian & SP
    print('--------------------NOISE--------------------')
    if(input.noiseType == 'gaussian'):
        noisyImage = corruptImage('gaussian', origImage, input.NoiseGMeanL, input.NoiseGSD, '')
    if(input.noiseType == 'sp'):
        noisyImage = corruptImage('sp', origImage, '', '', float(input.NoiseStrength))

    # Converting color images to selected single color spectrum
    if(input.SingleColorSpectum == 'R'):
        convertToSingleColorSpectrum(origImage, 'R')
    if(input.SingleColorSpectum == 'G'):
        convertToSingleColorSpectrum(origImage, 'G')
    if(input.SingleColorSpectum == 'B'):
        convertToSingleColorSpectrum(origImage, 'B')
    
    # Histogram calculation for each individual image
    print('--------------------HISTOGRAM & EQUALIZE HISTOGRAM--------------------')
    calc_histogram(origImage)


    # Selected image quantization technique for user-specified levels
    print('--------------------IMAGE QUANTIZATION--------------------')
    image_quantization(origImage, float(input.ImageQuantLevel))

    # Linear filter with user-specified mask size and pixel weights
    print('--------------------FILTERING OPERATIONS--------------------')
    linearFilterGaussian(noisyImage, 3, 1.5)

    final(entry)
    
def calc_histogram(image):
    # https://stackoverflow.com/questions/22159160/python-calculate-histogram-of-image
    # https://stackoverflow.com/questions/40700501/how-to-calculate-mean-color-of-image-in-numpy-array
    # https://matplotlib.org/2.0.2/users/image_tutorial.html
    # https://github.com/lxcnju/histogram_equalization/blob/master/contrast.py
    start_time = time.time()
    vals = np.mean(image, axis=(0, 1)).flatten()
    # bins are defaulted to image.max and image.min values
    hist, bins = np.histogram(vals, density=True)
    print("vals {}",format(vals))
    print("bins {}",format(bins))
    print("hist {}",format(hist))
    histSum = hist.sum()
    print('Histogram Sum {}'.format(histSum)) 
    # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html?highlight=histogram%20sum
    print('Cumulative Density Result {}'.format(np.sum(hist * np.diff(bins)))) 

    # plot histogram centered on values 0..255
    # plt.bar(bins[:-1] - 0.5, hist, width=1, edgecolor='none')
    """ plt.hist(vals, 256, range=(0.0, 1.0), fc='k', ec='k')
    plt.show()
    plt.bar(bins[:-1], hist, width=1, edgecolor='none')
    plt.xlim([-0.5, 255.5])
    plt.show() """

    eqHistogram = equalize_histogram(image, hist, bins)
    print('Equalize Histogram {}'.format(eqHistogram))
    end_time = (time.time() - start_time) % 60
    imageHistogramPt.append(end_time)
    return hist

def equalize_histogram(a, hist, bins):
    # https://gist.github.com/TimSC/6f429dfacf523f5c9a58c3b629f0540e
	""" a = np.array(a)
	hist, bins2 = np.histogram(a, bins=bins) """
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
    start_time = time.time()
    # https://stackoverflow.com/questions/38152081/how-do-you-quantize-a-simple-input-using-python - TODO
    result =  level * np.round(image/level) 
    print('Result {}'.format(result))
    end_time = (time.time() - start_time) % 60
    imageQuantizationPt.append(end_time)

def convertToSingleColorSpectrum(orig3DImage, colorSpectrum):
    start_time = time.time()
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
    end_time = (time.time() - start_time) % 60
    imageSingleSpectrumPt.append(end_time)

def pltImage(image, title):
    plt.ylabel('Height {}'.format(image.shape[0])) 
    plt.xlabel('Width {}'.format(image.shape[1])) 
    plt.title(title) 
    plt.axis('off')
    plt.imshow(image)
    # plt.show()

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

def corruptImage(noise_typ, image, mean, sd, strength):
    start_time = time.time()
    row,col,ch= image.shape
    if noise_typ == "gaussian":
        mean = mean
        sd = sd
        sigma = sd**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        print('>>>>>>>>>> Gaussian >>>>>>>>>>') 
        print(format(noisy)) 
        
    elif noise_typ == "sp":
        strength = strength
        amount = 0.004
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * strength)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        # out[coords] = 1
        noisy[tuple(coords)]
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - strength))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        noisy[coords] = 0
        print('>>>>>>>>>> Salt & Pepper >>>>>>>>>>') 
        #print(format(sp)) 
    
    end_time = (time.time() - start_time) % 60
    imageNoisyPt.append(end_time)
    return noisy

def linearFilterGaussian(noisyImage, maskSize=5, sigma=1.):
    start_time = time.time()
    # converti to 2D gray image first
    gray = rgb2gray(noisyImage)
    """ print("--------------------2D - GRAY--------------------")
    print("Size of the image array: ", gray.size)
    print('Shape of the image : {}'.format(gray.shape)) 
    print('Image Height {}'.format(gray.shape[0])) 
    print('Image Width {}'.format(gray.shape[1])) 
    print('Dimension of Image {}'.format(gray.ndim)) """

    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    # https://stackoverflow.com/questions/47369579/how-to-get-the-gaussian-filter
    # https://github.com/joeiddon/rpi_vision/blob/master/test.py
    # https://www.google.com/search?q=apply+gaussian+filter+to+image+%2B+numoy&oq=apply+gaussian+filter+to+image+%2B+numoy&aqs=chrome..69i57j0i333.12931j0j1&sourceid=chrome&ie=UTF-8
    # https://stackoverflow.com/questions/29920114/how-to-gauss-filter-blur-a-floating-point-numpy-array
    kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(maskSize-1)/2)**2+(y-(maskSize-1)/2)**2))/(2*sigma**2)), (maskSize, maskSize))
    result = kernel / np.sum(kernel)
    
    a = np.apply_along_axis(lambda x: np.convolve(x, result.flatten(), mode='same'), 0, gray)
    a = np.apply_along_axis(lambda x: np.convolve(x, result.flatten(), mode='same'), 1, gray)
    print("linearFilterGaussian: ", a)
    pltImage(a, 'Linear Filter')
    end_time = (time.time() - start_time) % 60
    imageLinearFilterPt.append(end_time)

def median_filter(imsga):
    # https://en.wikipedia.org/wiki/Kernel_(image_processing)
    # https://github.com/ijmbarr/image-processing-with-numpy/blob/master/image-processing-with-numpy.ipynb
    # https://github.com/susantabiswas/Digital-Image-Processing/blob/master/Day3/median_filter.py
    # https://stackoverflow.com/questions/58154630/image-smoothing-using-median-filter
    return ''

def final(entry):
    perf_metrics()
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

def read_input(input):
    with open('input.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                print('Path: {0}, NoiseType: {1}, NoiseStrength: {2}, NoiseGMeanL: {3}, NoiseGSD: {4}, SingleColorSpectum: {5}, ImageQuantLevel: {6}'.format(row[0],row[1],row[2],row[3],row[4],row[5],row[6]))
                #inp = InputFromCsv(row[0],row[1],row[2],row[3],row[4],row[5],row[6])  
                inp = getInput(row[0],row[1],row[2],row[3],row[4],row[5],row[6])
                process_batch(path, inp)
                line_count += 1
    print(f'Processed {line_count} lines.')

print('----------IMAGE ANALYSIS-------------------')
path = input('Enter images relative path: ')
if(path == '') :
    path = './Cancerouscellsmears2'
input = read_input('./input.txt')

basepath = process_batch(path)







       

        
       
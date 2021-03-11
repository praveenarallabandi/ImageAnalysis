# imports
import os
import toml
import numpy as np  
from typing import List
import matplotlib.pyplot as plt
import time
from PIL import Image # Used only for importing and exporting images

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
imageQuantizationMsePt = []
imageLinearFilterPt = []
imageMedianFilterPt = []

columnar = []
parabasal = []
intermediate = []
superficial = []
mild = []
severe = []


# Process files in directory as a batch
def process_batch(input):
    # basepath = ('./Cancerouscellsmears2')
    base_path = conf["DATA_DIR"]
    with os.scandir(base_path) as entries:
        groupImageClass(entries)

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

    for imageClass in imageClasses:
        for image in imageClasses[imageClass]:
            print('Processing Image - {}'.format(image.name))
            print('Processing Image Class - {}'.format(imageClass))
            # print('Processing Input - {0}, {1}, {2}, {3}, {4}, {5}, {6}'.format(input.path, input.noiseType, input.NoiseStrength, input.NoiseGMeanL, input.NoiseGSD, input.SingleColorSpectum, input.ImageQuantLevel))
            process_image(image, imageClass)
        imageClassesProcessTime[imageClass] = (time.time() - start_time) % 60

    perf_metrics()

def perf_metrics():
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
    ans = sum(imageQuantizationMsePt)
    avg = ans / len(imageQuantizationMsePt)
    print('{0} {1} \t {2}'.format('Quantization - MSE', ans, avg))
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
def process_image(entry, imageClass):
    try:
        # Given images is 1D array
        # origImage = np.fromfile(entry, dtype = np.uint8, count = TotalPixels)
        # origImage = plt.imread(conf["DATA_DIR"] + '/' + entry.name)
        origImage = np.asarray(Image.open(conf["DATA_DIR"] + entry.name))
        print("--------------------ORIGINAL IMAGE--------------------")
        print("Size of the image array: ", origImage.size)
        print('Type of the image : ' , type(origImage)) 
        print('Shape of the image : {}'.format(origImage.shape)) 
        print('Image Height {}'.format(origImage.shape[0])) 
        print('Image Width {}'.format(origImage.shape[1]))
        print('Dimension of Image {}'.format(origImage.ndim))
        pltImage(origImage, 'Original Image')

        # Converting color images to selected single color spectrum
        singleSpectrumImage = convertToSingleColorSpectrum(origImage, conf["COLOR_CHANNEL"])

        print("--------------------singleSpectrumImage IMAGE--------------------")
        print("Size of the image array: ", singleSpectrumImage.size)
        print('Type of the image : ' , type(singleSpectrumImage)) 
        print('Shape of the image : {}'.format(singleSpectrumImage.shape)) 
        print('Image Height {}'.format(singleSpectrumImage.shape[0])) 
        print('Image Width {}'.format(singleSpectrumImage.shape[1]))
        print('Dimension of Image {}'.format(singleSpectrumImage.ndim))

        # Noise addition functions that will allow to corrupt each image with Gaussian & SP
        print('--------------------NOISE--------------------')
        noisyGaussianImage = corruptImageGaussian(singleSpectrumImage, conf["GAUSS_NOISE_STRENGTH"])
        noisySaltPepperImage = corruptImageSaltAndPepper(singleSpectrumImage, conf["SALT_PEPPER_STRENGTH"])
        
        # Histogram calculation for each individual image
        print('--------------------HISTOGRAM & EQUALIZE HISTOGRAM--------------------')
        histogram, eqHistogram, eqImage = calc_histogram(singleSpectrumImage)


        # Selected image quantization technique for user-specified levels
        print('--------------------IMAGE QUANTIZATION--------------------')
        quantImage = image_quantization(singleSpectrumImage, conf["IMAGE_QUANT_LEVELS"])

        # Selected image quantization technique for user-specified levels
        print('--------------------IMAGE QUANTIZATION MEAN SQUARE ERROR (MSE)--------------------')
        image_quantization_mse(singleSpectrumImage, eqImage)

        # Linear filter with user-specified mask size and pixel weights
        print('--------------------FILTERING OPERATIONS--------------------')
        linear = linearFilter(singleSpectrumImage, conf["LINEAR_MASK"], conf["LINEAR_WEIGHTS"])
        median = medianFilter(singleSpectrumImage, conf["MEDIAN_MASK"], conf["MEDIAN_WEIGHTS"])
        # mean_filter(noisyImage, 3, 2)
        # median_filter(noisyImage, 2, 2, 2)

        export_image(noisySaltPepperImage, "salt_and_pepper_" + entry.name)
        export_image(noisyGaussianImage, "gaussian_" + entry.name)
        export_image(eqImage, "equalized_" + entry.name)
        export_image(linear, "linear_" + entry.name)
        export_image(median, "median_" + entry.name)

        export_plot(histogram, "histogram_" + entry.name)
        export_plot(eqHistogram, "eqhistogram_" + entry.name)

        # final(entry)

    except Exception as e:
        print(e)
        return e
    
def histogram(image: np.array, bins) -> np.array:
    vals = np.mean(image, axis=(0,1)).flatten()
    # bins are defaulted to image.max and image.min values
    hist, bins2 = np.histogram(vals, bins, density=True)
    return hist
    
# CALCULATE HISTOGRAM    
def calc_histogram(image):
    # https://stackoverflow.com/questions/22159160/python-calculate-histogram-of-image
    # https://stackoverflow.com/questions/40700501/how-to-calculate-mean-color-of-image-in-numpy-array
    # https://matplotlib.org/2.0.2/users/image_tutorial.html
    # https://github.com/lxcnju/histogram_equalization/blob/master/contrast.py
    start_time = time.time()
    maxval = 255.0
    bins = np.linspace(0.0, maxval, 257)
    hist = histogram(image, bins)
    equalized = equalize_histogram(image, hist, bins)
    imgEqualized = np.reshape(equalized, image.shape)
    end_time = (time.time() - start_time) % 60
    imageHistogramPt.append(end_time)
    return hist, histogram(equalized, bins), imgEqualized

# EQUALIZE HISTOGRAM
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

# IMAGE QUANTIZATION
def image_quantization(image, level):
    start_time = time.time()
    # https://stackoverflow.com/questions/38152081/how-do-you-quantize-a-simple-input-using-python - TODO
    print(image)
    result =  level * np.round(image/level) 
    print('Result {}'.format(result))
    end_time = (time.time() - start_time) % 60
    imageQuantizationPt.append(end_time)
    return result

def image_quantization_mse(image, imageQuant):
    start_time = time.time()

    mse = (np.square(image - imageQuant)).mean(axis=None)
    print('MSE {}'.format(mse))
    end_time = (time.time() - start_time) % 60
    imageQuantizationMsePt.append(end_time)

def convertToSingleColorSpectrum(orig3DImage, colorSpectrum):
    start_time = time.time()
    if(colorSpectrum == 'R') :
        return orig3DImage[:, :, 0]
        
    if(colorSpectrum == 'G') :
        return orig3DImage[:, :, 1]

    if(colorSpectrum == 'B') :
        return orig3DImage[:, :, 2]

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

def corruptImageGaussian(image, strength):
    start_time = time.time()
    mean = 0.0
    noise = np.random.normal(mean,strength,image.size)
    reshaped_noise = noise.reshape(image.shape)
    gaussian = image + reshaped_noise
    print('>>>>>>>>>> Gaussian >>>>>>>>>>') 
    print(format(gaussian))
    
    end_time = (time.time() - start_time) % 60
    imageNoisyPt.append(end_time)
    return gaussian

def corruptImageSaltAndPepper(image, strength):
    start_time = time.time()
        
    s_vs_p = 0.5
    noisy = np.copy(image)

    # Generate Salt '1' noise
    num_salt = np.ceil(strength * image.size * s_vs_p)

    for i in range(int(num_salt)):
        x = np.random.randint(0, image.shape[0] - 1)
        y = np.random.randint(0, image.shape[1] - 1)
        noisy[x][y] = 0

    # Generate Pepper '0' noise
    num_pepper = np.ceil(strength * image.size * (1.0 - s_vs_p))

    for i in range(int(num_pepper)):
        x = np.random.randint(0, image.shape[0] - 1)
        y = np.random.randint(0, image.shape[1] - 1)
        noisy[x][y] = 0
        
    print('>>>>>>>>>> Salt & Pepper >>>>>>>>>>') 
    print(format(noisy))
         
    
    end_time = (time.time() - start_time) % 60
    imageNoisyPt.append(end_time)
    return noisy

def apply_filter(img_array: np.array, img_filter: np.array) -> np.array:
    """
    Applies a linear filter to a copy of an image based on filter weights
    """

    rows, cols = img_array.shape
    height, width = img_filter.shape

    output = np.zeros((rows - height + 1, cols - width + 1))

    for rr in range(rows - height + 1):
        for cc in range(cols - width + 1):
            for hh in range(height):
                for ww in range(width):
                    imgval = img_array[rr + hh, cc + ww]
                    filterval = img_filter[hh, ww]
                    output[rr, cc] += imgval * filterval

    return output

def linearFilter(noisyImage, maskSize=5, weights = List[List[int]]):
    start_time = time.time()
    print('>>>>>>>>>>LINEAR<<<<<<<<<<<<<<<')
    filter = np.array(weights)
    linear = apply_filter(noisyImage, filter)

    end_time = (time.time() - start_time) % 60
    imageLinearFilterPt.append(end_time)
    print(linear)
    return linear

""" def meanFilter(image, maskSize, weights):
    # Set the kernel.
    height, width = image.shape[:2]
    kernel = np.ones((maskSize, maskSize), np.float32) / weights

    for row in range(1, height + 1):
        for column in range(1, width + 1):
            # Get the area to be filtered with range indexing.
            filter_area = image[row - 1:row + 2, column - 1:column + 2]
            res = np.sum(np.multiply(kernel, filter_area))
            image[row][column] = res

    print('Mean Filter - {0}'.format(image))
    return image

def median_filter(image, maskSize, weights):
    # Set the kernel.
    height, width = image.shape[:2]

    for row in range(1, height + 1):
        for column in range(1, width + 1):
            filter_area = image[row - 1:row + 2, column - 1:column + 2]
            image[row][column] = np.median(filter_area)

    print('Median Filter - {0}'.format(image))
    return image """

def apply_median_filter(img_array: np.array, img_filter: np.array) -> np.array:
    """
    Applies a linear filter to a copy of an image based on filter weights
    """

    rows, cols = img_array.shape
    height, width = img_filter.shape

    pixel_values = np.zeros(img_filter.size ** 2)
    output = np.zeros((rows - height + 1, cols - width + 1))

    for rr in range(rows - height + 1):
        for cc in range(cols - width + 1):

            p = 0
            for hh in range(height):
                for ww in range(width):

                    pixel_values[p] = img_array[hh][ww]
                    p += 1

            # Sort the array of pixels inplace
            pixel_values.sort()

            # Assign the median pixel value to the filtered image.
            output[rr][cc] = pixel_values[p // 2]

    return output

def medianFilter(noisyImage, maskSize=5, weights = List[List[int]]):
    # https://en.wikipedia.org/wiki/Kernel_(image_processing)
    # https://github.com/ijmbarr/image-processing-with-numpy/blob/master/image-processing-with-numpy.ipynb
    # https://github.com/susantabiswas/Digital-Image-Processing/blob/master/Day3/median_filter.py
    # https://stackoverflow.com/questions/58154630/image-smoothing-using-median-filter
    print('>>>>>>>>>>MEDIAN<<<<<<<<<<<<<<<')
    filter = np.array(weights)
    median = apply_median_filter(noisyImage, filter)
    print(median)
    return median

def final(entry):
    perf_metrics()
    entry.close()
    test.close()
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

def export_image(image: np.array, filename: str) -> None:
    """
    Exports a numpy array as a grey scale bmp image
    """
    print('Processing {0}'.format(filename))
    img = Image.fromarray(image)
    img = img.convert("L")
    img.save(conf["OUTPUT_DIR"] + filename)

def export_plot(image: np.array, filename: str) -> None:
    """
    exports a historgam as a matplotlib plot
    """

    _ = plt.hist(image, bins=256, range=(0, 256))
    plt.title(filename)
    plt.savefig(conf["OUTPUT_DIR"] + filename + ".png")
    plt.close()

""" def read_input(input):
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
    print(f'Processed {line_count} lines.') """

print('----------IMAGE ANALYSIS-------------------')
""" # path = input('Enter images relative path: ')
if(path == '') :
    path = './Cancerouscellsmears2' """
global conf
conf = toml.load('./config.toml')
# input = read_input('./input.txt')

basepath = process_batch(conf)







       

        
       
# imports
import os
import toml
import numpy as np  
import numba as nb
from typing import List
import matplotlib.pyplot as plt
import time
from PIL import Image # Used only for importing and exporting images

start_time = time.time()

# resolution for images
ROWS = 768    
COLS =  568
TotalPixels = ROWS * COLS
imageClasses = {}
imageClassesProcessTime = {}
temp = {}
imageNoisyGaussianPt = []
imageHistogramPt = []
imageSingleSpectrumPt = []
imageNoisySaltPepperPt = []
imageQuantizationMsePt = []
imageLinearFilterPt = []
imageMedianFilterPt = []

columnar = []
parabasal = []
intermediate = []
superficial = []
mild = []
severe = []

# timeit: decorator to time functions


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        end_time = (end_time - start_time) * 1000
        if(f.__name__ == 'convertToSingleColorSpectrum'):
            imageSingleSpectrumPt.append(end_time)
        if(f.__name__ == 'corruptImageGaussian'):
            imageNoisyGaussianPt.append(end_time)
        if(f.__name__ == 'corruptImageSaltAndPepper'):
            imageNoisySaltPepperPt.append(end_time)
        if(f.__name__ == 'calculateHistogram'):
            imageHistogramPt.append(end_time)
        if(f.__name__ == 'linearFilter'):
            imageLinearFilterPt.append(end_time)
        if(f.__name__ == 'medianFilter'):
            imageMedianFilterPt.append(end_time)
        # single_time_data[f.__name__] = (end_time - start_time) * 1000
        # print((end_time - start_time) * 1000)
        return result

    return timed

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
            # print('Processing Image - {}'.format(image.name))
            # print('Processing Input - {0}, {1}, {2}, {3}, {4}, {5}, {6}'.format(input.path, input.noiseType, input.NoiseStrength, input.NoiseGMeanL, input.NoiseGSD, input.SingleColorSpectum, input.ImageQuantLevel))
            process_image(image, imageClass)

    perf_metrics()

def perf_metrics():
    print('********************************************************************')
    print('\t\t PERFORMANCE METRICS ')
    print('********************************************************************')
    print('--------------------------------------------------------------------')
    print('Total Processig time: {} sec'.format(time.time() - start_time))
    print('--------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    print('Procedure \t Total Execution Time (ms) \t Average Time Per Image (ms)')
    print('-----------------------------------------------------------------------')
    ans = sum(imageNoisyGaussianPt)
    avg = ans / len(imageNoisyGaussianPt)
    print('{0} \t {1} \t {2}'.format('Gaussian Noise', ans, avg))
    ans = sum(imageNoisySaltPepperPt)
    avg = ans / len(imageNoisySaltPepperPt)
    print('{0} \t {1} \t {2}'.format('Salt & Pepper', ans, avg))
    ans = sum(imageHistogramPt)
    avg = ans / len(imageHistogramPt)
    print('{0} \t {1} \t {2}'.format('Histogram', ans, avg))
    ans = sum(imageSingleSpectrumPt)
    avg = ans / len(imageSingleSpectrumPt)
    print('{0}  {1} \t {2}'.format('Single Spectrum', ans, avg))
    ans = sum(imageLinearFilterPt)  
    avg = ans / len(imageLinearFilterPt)
    print('{0} \t {1} \t {2}'.format('Linear Filter', ans, avg))
    ans = sum(imageMedianFilterPt)
    avg = ans / len(imageMedianFilterPt)
    print('{0} \t {1} \t {2}'.format('Median Filter', ans, avg))
    """ print('-----------------------------------------------------------------------')
    for classTime in imageClassesProcessTime:
        print('Processig time for {0}: \t\t {1} sec'.format(classTime, imageClassesProcessTime[classTime]))
    print('-----------------------------------------------------------------------') """

# Process the input image
def process_image(entry, imageClass):
    single_time_data = {}
    try:
        start_time = time.time()
        origImage = np.asarray(Image.open(conf["DATA_DIR"] + entry.name))

        # Converting color images to selected single color spectrum
        singleSpectrumImage = timeit(convertToSingleColorSpectrum)(
            origImage, conf["COLOR_CHANNEL"]
        )


        # Noise addition functions that will allow to corrupt each image with Gaussian & SP
        # print('--------------------NOISE--------------------')
        # noisyGaussianImage = corruptImageGaussian(singleSpectrumImage, conf["GAUSS_NOISE_STRENGTH"])
        noisyGaussianImage = timeit(corruptImageGaussian)(singleSpectrumImage, conf["GAUSS_NOISE_STRENGTH"])
        # noisySaltPepperImage = corruptImageSaltAndPepper(singleSpectrumImage, conf["SALT_PEPPER_STRENGTH"])
        noisySaltPepperImage = timeit(corruptImageSaltAndPepper)(singleSpectrumImage, conf["SALT_PEPPER_STRENGTH"])
        
        # Histogram calculation for each individual image
        # print('--------------------HISTOGRAM, EQUALIZE HISTOGRAM & IMAGE QUANTIZATION--------------------')
        # histogram, eqHistogram, quantizedImage = calculateHistogram(singleSpectrumImage)
        histogram, eqHistogram, quantizedImage = timeit(calculateHistogram)(singleSpectrumImage)

        # Linear filter with user-specified mask size and pixel weights
        # print('--------------------FILTERING OPERATIONS--------------------')
        # linear = linearFilter(singleSpectrumImage, conf["LINEAR_MASK"], conf["LINEAR_WEIGHTS"])
        linear = timeit(linearFilter)(singleSpectrumImage, conf["LINEAR_MASK"], conf["LINEAR_WEIGHTS"])
        # median = medianFilter(singleSpectrumImage, conf["MEDIAN_MASK"], conf["MEDIAN_WEIGHTS"])
        median = timeit(medianFilter)(singleSpectrumImage, conf["MEDIAN_MASK"], conf["MEDIAN_WEIGHTS"])

        exportImage(noisySaltPepperImage, "salt_and_pepper_" + entry.name)
        exportImage(noisyGaussianImage, "gaussian_" + entry.name)
        exportImage(quantizedImage, "equalized_" + entry.name)
        exportImage(linear, "linear_" + entry.name)
        exportImage(median, "median_" + entry.name)

        exportPlot(histogram, "histogram_" + entry.name)
        exportPlot(eqHistogram, "eqhistogram_" + entry.name)

        # Selected image quantization technique for user-specified levels
        # print('--------------------IMAGE QUANTIZATION MEAN SQUARE ERROR (MSE)--------------------')
        image_quantization_mse(singleSpectrumImage, quantizedImage, entry.name)

        end_time = time.time()
        single_time_data["total"] = (end_time - start_time) * 1000
    except Exception as e:
        print(e)
        return e
    
def histogram(image: np.array, bins) -> np.array:
    """Calculate histogram for specified image

    Args:
        image (np.array): input image
        bins ([type]): number of bins

    Returns:
        np.array: calculated histogram value
    """
    vals = np.mean(image, axis=0)
    # bins are defaulted to image.max and image.min values
    hist, bins2 = np.histogram(vals, bins, density=True)
    return hist
    
# CALCULATE HISTOGRAM    
def calculateHistogram(image):
    """Calculate histogram, Equalized image hostogram and quantized image for specified image

    Args:
        image ([type]): input image

    Returns:
        [type]: Histogram, Equalized Histogram , Quantized Image
    """
    # start_time = time.time()
    maxval = 255.0
    bins = np.linspace(0.0, maxval, 257)
    flatImage = image.flatten()
    hist = histogram(flatImage, bins)
    equalized = equalize_histogram(flatImage, hist, bins)
    imgEqualized = np.reshape(equalized, image.shape)
    """ end_time = (time.time() - start_time) % 60
    imageHistogramPt.append(end_time) """
    return hist, histogram(equalized, bins), imgEqualized

# EQUALIZE HISTOGRAM
def equalize_histogram(image, hist, bins):
    cumsum = np.cumsum(hist)
    nj = (cumsum - cumsum.min()) * 255
    N = cumsum.max() - cumsum.min()
    cumsum = nj / N
    casted = cumsum.astype(np.uint8)
    equalized = casted[image]
    return equalized

def image_quantization_mse(image, imageQuant, imageName):
    """Calculate MSE between original inage and quantized image

    Args:
        image ([type]): imput image
        imageQuant ([type]): quantized image
        imageName ([type]): original image
    """
    start_time = time.time()
    
    mse = (np.square(image - imageQuant)).mean(axis=None)
    print('<{0}> Completed Execution - MSE: {1}'.format(imageName, mse))
    end_time = (time.time() - start_time) % 60
    imageQuantizationMsePt.append(end_time)

def convertToSingleColorSpectrum(orig3DImage, colorSpectrum):
    """Get the image based on R, G or B specturm

    Args:
        orig3DImage ([type]): original image
        colorSpectrum ([type]): color specturm

    Returns:
        [type]: image for single color spectrum
    """
    """ start_time = time.time() """

    if(colorSpectrum == 'R') :
        img = orig3DImage[:, :, 0]
        
    if(colorSpectrum == 'G') :
        img = orig3DImage[:, :, 1]

    if(colorSpectrum == 'B') :
        img = orig3DImage[:, :, 2]

    """ end_time = (time.time() - start_time) % 60
    print(end_time)
    imageSingleSpectrumPt.append(end_time) """
    return img

def corruptImageGaussian(image, strength):
    """Apply gaussian with user specified strength

    Args:
        mage ([type]): input image
        strength ([type]): user specified strength

    Returns:
        [type]: Gaussian applied noisy image
    """
    # start_time = time.time()
    mean = 0.0
    noise = np.random.normal(mean,strength,image.size)
    reshaped_noise = noise.reshape(image.shape)
    gaussian = image + reshaped_noise
    
    """ end_time = (time.time() - start_time) % 60
    imageNoisyGaussianPt.append(end_time) """
    return gaussian

def corruptImageSaltAndPepper(image, strength):
    """Apply salt and pepper with user specified strength

    Args:
        image ([type]): input image
        strength ([type]): user specified strength

    Returns:
        [type]: Salt & Perpper applied noisy image
    """
    s_vs_p = 0.5
    noisy = np.copy(image)

    # Generate Salt '1' noise
    num_salt = np.ceil(strength * image.size * s_vs_p)
    cords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[tuple(cords)] = 1

    # Generate Pepper '0' noise
    num_pepper = np.ceil(strength * image.size * (1.0 - s_vs_p))
    cords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[tuple(cords)] = 0
    return noisy

@nb.njit(fastmath=True)
def applyFilter(image: np.array, weightArray: np.array) -> np.array:
    """Applying the filter loops through every pixel in the image and 
    multiples the neighboring pixel values by the weights in the kernel.

    Args:
        image (np.array): `[description]`
        weightArray (np.array): [description]

    Returns:
        np.array: new filter applied array
    """

    rows, cols = image.shape
    height, width = weightArray.shape
    output = np.zeros((rows - height + 1, cols - width + 1))
    rowRange = np.arange(rows - height + 1)
    colRange = np.arange(cols - width + 1)
    heightRange = np.arange(height)
    widthRange = np.arange(width)

    for rrow in rowRange:
        for ccolumn in colRange:
            for hheight in heightRange:
                for wwidth in widthRange:
                    imgval = image[rrow + hheight, ccolumn + wwidth]
                    filterval = weightArray[hheight, wwidth]
                    output[rrow, ccolumn] += imgval * filterval
                    
    return output

def linearFilter(image, maskSize=9, weights = List[List[int]]):
    """Linear filetering

    Args:
        image ([type]): Image on filetering is applied
        maskSize (int, optional): mask size. Defaults to 9.
        weights ([type], optional): User defined weights that are applied to each pi. Defaults to List[List[int]].

    Returns:
        [type]: [description]
    """
    # start_time = time.time()
    # print('>>>>>>>>>>LINEAR<<<<<<<<<<<<<<<')
    filter = np.array(weights)
    linear = applyFilter(image, filter)

    """ end_time = (time.time() - start_time) % 60
    imageLinearFilterPt.append(end_time)
    print(linear) """
    return linear


def applyMedianFilter(image: np.array, filter: np.array) -> np.array:
    rows, cols = image.shape
    height, width = filter.shape

    pixels = np.zeros(filter.size ** 2)
    output = np.zeros((rows - height + 1, cols - width + 1))

    for rrows in range(rows - height + 1):
        for ccolumns in range(cols - width + 1):

            p = 0
            for hheight in range(height):
                for wweight in range(width):

                    pixels[p] = image[hheight][wweight]
                    p += 1

            # Sort the array of pixels inplace
            pixels.sort()

            # Assign the median pixel value to the filtered image.
            output[rrows][ccolumns] = pixels[p // 2]

    return output

def medianFilter(image, maskSize=9, weights = List[List[int]]):
    """Median filter - Apply the median filter to median pixel value of neghbourhood. 
    Applies a linear filter to a copy of an image based on filter weights

    Args:
        image ([type]): inout image
        maskSize (int, optional): [description]. Defaults to 9.
        weights ([type], optional): user specified weights. Defaults to List[List[int]].

    Returns:
        [type]: [description]
    """
    # start_time = time.time()
    # print('>>>>>>>>>>MEDIAN<<<<<<<<<<<<<<<')
    filter = np.array(weights)
    median = applyMedianFilter(image, filter)
    """ end_time = (time.time() - start_time) % 60
    imageMedianFilterPt.append(end_time)
    print(median) """
    return median

def exportImage(image: np.array, filename: str) -> None:
    """export image to specified location

    Args:
        image (np.array): image to export
        filename (str): file name to create
    """
    img = Image.fromarray(image)
    img = img.convert("L")
    img.save(conf["OUTPUT_DIR"] + filename)

def exportPlot(image: np.array, filename: str) -> None:
    """exports a historgam as a matplotlib plot

    Args:
        image (np.array): image to export
        filename (str): file name to create
    """
    _ = plt.hist(image, bins=256, range=(0, 256))
    plt.title(filename)
    plt.savefig(conf["OUTPUT_DIR"] + filename + ".png")
    plt.close()

def main():
    print('----------IMAGE ANALYSIS START-------------------')
    global conf
    conf = toml.load('./config.toml')

    process_batch(conf)

if __name__ == "__main__":
    main()







       

        
       

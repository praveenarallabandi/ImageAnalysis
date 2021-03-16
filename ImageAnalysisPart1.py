# imports
import os
import toml
import numpy as np  
import numba as nb
from typing import List
import matplotlib.pyplot as plt
import time
import os
from colorama import Fore, Style
from PIL import Image # Used only for importing and exporting images

total_start_time = time.time()

# resolution for images
ROWS = 768    
COLS =  568
TotalPixels = ROWS * COLS
imageClasses = {}
imageClassesProcessTime = {}
temp = {}
trackMse = {}
imageNoisyGaussianPt = []
imageHistogramPt = []
imageSingleSpectrumPt = []
imageNoisySaltPepperPt = []
imageQuantizationMsePt = []
imageLinearFilterPt = []
imageMedianFilterPt = []
imageExport = []
plotExport = []

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
        if(f.__name__ == 'image_quantization_mse'):
            imageQuantizationMsePt.append(end_time)
        if(f.__name__ == 'exportImage'):
            imageExport.append(end_time)
        if(f.__name__ == 'exportPlot'):
            plotExport.append(end_time)
        return result

    return timed

# Process files in directory as a batch
def process_batch(input):
    base_path = conf["INPUT_DIR"]
    with os.scandir(base_path) as entries:
        groupImageClass(entries)

def groupImageClass(entries):
    """Group images into class based on their name

    Args:
        entries ([type]): Batch files in directory
    """
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
            process_image(image, imageClass)

    perf_metrics()

def printMsg(msg: str, avg, ans):
    print(f'{Fore.YELLOW}{msg}{Style.RESET_ALL} \t{Fore.BLUE} {avg} \t {ans} {Style.RESET_ALL}')

def perf_metrics():
    print('Processing completed!')
    for mse in trackMse:
        print('<{0}> Completed Execution - MSE: {1}'.format(mse, trackMse[mse]))
    
    print('********************************************************************')
    print(f'\t\t {Fore.CYAN}PERFORMANCE METRICS {Style.RESET_ALL}')
    print('********************************************************************')
    print('-----------------------------------------------------------------------')
    print(f'{Fore.CYAN}Procedure \t Average Per Image (ms)  Total Execution Time (ms){Style.RESET_ALL}')
    print('-----------------------------------------------------------------------')
    totalAvg = 0.0
    totalAns = 0.0
    RED = "\x1b[1;31;40m"
    ans = sum(imageNoisyGaussianPt)
    avg = ans / len(imageNoisyGaussianPt)
    totalAvg = totalAvg + avg
    # print(f'{Fore.YELLOW}Gaussian Noise{Style.RESET_ALL} \t{Fore.BLUE} {avg} \t {ans} {Style.RESET_ALL}')
    printMsg('Gaussian Noise', avg, ans)
    ans = sum(imageNoisySaltPepperPt)
    avg = ans / len(imageNoisySaltPepperPt)
    totalAvg = totalAvg + avg
    totalAns = totalAns + ans
    # print('{0} \t {1} \t {2}'.format('Salt & Pepper', avg, ans))
    printMsg('Salt & Pepper', avg, ans)
    ans = sum(imageHistogramPt)
    avg = ans / len(imageHistogramPt)
    totalAvg = totalAvg + avg
    totalAns = totalAns + ans
    # print('{0} \t {1} \t {2}'.format('Histogram', avg, ans))
    printMsg('Histogram', avg, ans)
    ans = sum(imageSingleSpectrumPt)
    avg = ans / len(imageSingleSpectrumPt)
    totalAvg = totalAvg + avg
    totalAns = totalAns + ans
    # print('{0}  {1} \t {2}'.format('Single Spectrum', avg, ans))
    # printMsg('Single Spectrum', avg, ans)
    print(f'{Fore.YELLOW}Single Spectrum{Style.RESET_ALL} {Fore.BLUE} {avg} \t {ans} {Style.RESET_ALL}')
    ans = sum(imageLinearFilterPt)  
    avg = ans / len(imageLinearFilterPt)
    totalAvg = totalAvg + avg
    totalAns = totalAns + ans
    # print('{0} \t {1} \t {2}'.format('Linear Filter', avg, ans))
    printMsg('Linear Filter', avg, ans)
    ans = sum(imageMedianFilterPt)
    avg = ans / len(imageMedianFilterPt)
    totalAvg = totalAvg + avg
    totalAns = totalAns + ans
    # print('{0} \t {1} \t {2}'.format('Median Filter', avg, ans))
    printMsg('Median Filter', avg, ans)
    # print('{0} \t {1}'.format('TOTAL \t', totalAvg))
    # printMsg('TOTAL \t', totalAvg, totalAns)
    print(f'{Fore.GREEN}TOTAL \t\t{Style.RESET_ALL} {Fore.CYAN}{totalAvg} \t {totalAns} {Style.RESET_ALL}')
    ans = sum(imageExport)
    avg = ans / len(imageExport)
    # print('{0} \t {1} \t {2}'.format('Export Image', avg, ans))
    printMsg('Export Image', avg, ans)
    ans = sum(plotExport)
    avg = ans / len(plotExport)
    # print('{0} \t {1} \t {2}'.format('Plot Image', avg, ans))
    printMsg('Plot Image', avg, ans)
    print('--------------------------------------------------------------------')
    print(f'{Fore.GREEN}Total Processig time:{Style.RESET_ALL} {Fore.GREEN}{time.time() - total_start_time} sec')
    print('--------------------------------------------------------------------')

# Process the input image
def process_image(entry, imageClass):
    """Process each image with specific procedures

    Args:
        entry ([type]): Input image
        imageClass ([type]): Image class which it belongs to

    Returns:
        [type]: [description]
    """
    try:
        origImage = np.asarray(Image.open(conf["INPUT_DIR"] + entry.name))

        # Converting color images to selected single color spectrum
        singleSpectrumImage = timeit(convertToSingleColorSpectrum)(origImage, conf["COLOR_CHANNEL"])
        
        # Noise addition functions that will allow to corrupt each image with Gaussian & SP
        # print('--------------------NOISE--------------------')
        noisyGaussianImage = timeit(corruptImageGaussian)(singleSpectrumImage, conf["GAUSS_NOISE_STRENGTH"])
        noisySaltPepperImage = timeit(corruptImageSaltAndPepper)(singleSpectrumImage, conf["SALT_PEPPER_STRENGTH"])
        
        # Histogram calculation for each individual image
        # print('--------------------HISTOGRAM, EQUALIZE HISTOGRAM & IMAGE QUANTIZATION--------------------')
        histogram, eqHistogram, quantizedImage = timeit(calculateHistogram)(singleSpectrumImage)

        # Linear filter with user-specified mask size and pixel weights
        # print('--------------------FILTERING OPERATIONS--------------------')
        linear = timeit(linearFilter)(singleSpectrumImage, conf["LINEAR_MASK"], conf["LINEAR_WEIGHTS"])
        median = timeit(medianFilter)(singleSpectrumImage, conf["MEDIAN_MASK"], conf["MEDIAN_WEIGHTS"])

        timeit(exportImage)(noisySaltPepperImage, "salt_and_pepper_" + entry.name)
        timeit(exportImage)(noisyGaussianImage, "noisyGaussianImage" + entry.name)
        timeit(exportImage)(quantizedImage, "equalized_" + entry.name)
        timeit(exportImage)(linear, "linear_" + entry.name)
        timeit(exportImage)(median, "median_" + entry.name)

        timeit(exportPlot)(histogram, "histogram_" + entry.name)
        timeit(exportPlot)(eqHistogram, "eqhistogram_" + entry.name)

        # Selected image quantization technique for user-specified levels
        # print('--------------------IMAGE QUANTIZATION MEAN SQUARE ERROR (MSE)--------------------')
        timeit(image_quantization_mse)(singleSpectrumImage, quantizedImage, entry.name)

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
    
def calculateHistogram(image: np.array):
    """Calculate histogram, Equalized image hostogram and quantized image for specified image

    Args:
        image ([type]): input image

    Returns:
        [type]: Histogram, Equalized Histogram , Quantized Image
    """
    maxval = 255.0
    bins = np.linspace(0.0, maxval, 257)
    flatImage = image.flatten()
    hist = histogram(flatImage, bins)
    equalized = equalize_histogram(flatImage, hist, bins)
    imgEqualized = np.reshape(equalized, image.shape)
    return hist, histogram(equalized, bins), imgEqualized

def equalize_histogram(image: np.array, hist: np.array, bins) -> np.array:
    cumsum = np.cumsum(hist)
    nj = (cumsum - cumsum.min()) * 255
    N = cumsum.max() - cumsum.min()
    cumsum = nj / N
    casted = cumsum.astype(np.uint8)
    equalized = casted[image]
    return equalized

def image_quantization_mse(image: np.array, imageQuant: np.array, imageName: str) -> float:
    """Calculate MSE between original inage and quantized image

    Args:
        image ([type]): imput image
        imageQuant ([type]): quantized image
        imageName ([type]): original image
    """
    
    mse = (np.square(image - imageQuant)).mean(axis=None)
    trackMse[imageName] = mse

def convertToSingleColorSpectrum(image: np.array, colorSpectrum: str) -> np.array:
    """Get the image based on R, G or B specturm

    Args:
        image ([type]): original image
        colorSpectrum ([type]): color specturm

    Returns:
        [type]: image for single color spectrum
    """
    if(colorSpectrum == 'R') :
        img = image[:, :, 0]
        
    if(colorSpectrum == 'G') :
        img = image[:, :, 1]

    if(colorSpectrum == 'B') :
        img = image[:, :, 2]

    return img

def corruptImageGaussian(image: np.array, strength: int) ->  np.array:
    """Apply gaussian with user specified strength

    Args:
        mage ([type]): input image
        strength ([type]): user specified strength

    Returns:
        [type]: Gaussian applied noisy image
    """
    mean = 0.0
    noise = np.random.normal(mean,strength,image.size)
    reshaped_noise = noise.reshape(image.shape)
    gaussian = np.add(image, reshaped_noise)
    return gaussian


def corruptImageSaltAndPepper(image: np.array, strength: int) -> np.array:
    """Apply salt and pepper with user specified strength

    Args:
        image ([type]): input image
        strength ([type]): user specified strength

    Returns:
        [type]: Salt & Perpper applied noisy image
    """
    svsp = 0.5
    noisy = np.copy(image)

    # Salt '1' noise
    num_salt = np.ceil(strength * image.size * svsp)
    cords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[tuple(cords)] = 1

    # Pepper '0' noise
    num_pepper = np.ceil(strength * image.size * (1.0 - svsp))
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

    for rrow in range(rows - height + 1):
        for ccolumn in range(cols - width + 1):
            for hheight in range(height): # Need to vectorize this
                for wwidth in range(width):
                    imgval = image[rrow + hheight, ccolumn + wwidth]
                    filterval = weightArray[hheight, wwidth]
                    output[rrow, ccolumn] += imgval * filterval           
    return output

""" def applyFilterMethod3(image: np.array, weightArray: np.array) -> np.array:
    rows, cols = image.shape 
    height, width = weightArray.shape
    output = np.zeros((rows - height + 1, cols - width + 1))
    print(range(rows - height + 1))
    rrange = range(rows - height + 1)
    print(range(cols - width + 1))
    crange = range(cols - width + 1)
    print(range(height))
    rindex = len(rrange)
    index = 1
    for hheight in range(height):
        for wwidth in range(width):
            print(index)
            #imgval = image[rrange[:1][0] + hheight, crange[index: len(crange)][0] + wwidth]
            imgval = image[rrange[:1][0] + hheight, crange[:1][0] + wwidth]
            # imgval = image[rrange[:1] + hheight, crange[:1] + wwidth]
            filterval = weightArray[hheight, wwidth]
            output[:, :] += imgval * filterval
            index = index + 1 
             
    return output """

def linearFilter(image, maskSize=9, weights = List[List[int]]) -> np.array:
    """Linear filtering

    Args:
        image ([type]): Image on filetering is applied
        maskSize (int, optional): mask size. Defaults to 9.
        weights ([type], optional): User defined weights that are applied to each pi. Defaults to List[List[int]].

    Returns:
        [type]: [description]
    """
    filter = np.array(weights)
    filter = filter/sum(sum(filter))
    linear = applyFilter(image, filter)
    return linear

def applyMedianFilter(image: np.array, filter: np.array) -> np.array:
    rows, cols = image.shape
    height, width = filter.shape

    pixels = np.zeros(filter.size ** 2)
    output = np.zeros((rows - height + 1, cols - width + 1))

    for rrows in range(rows - height + 1):
        for ccolumns in range(cols - width + 1):
            index = 0
            for hheight in range(height): # Need to vectorize this
                for wweight in range(width):
                    pixels[index] = image.item(hheight, wweight)
                    index += 1
                    pixels.sort()
                    output[rrows][ccolumns] = pixels[index // 2]

    return output

def medianFilter(image, maskSize=9, weights = List[List[int]]):
    """Median filter - Apply the median filter to median pixel value of neghbourhood.

    Args:
        image ([type]): inout image
        maskSize (int, optional): [description]. Defaults to 9.
        weights ([type], optional): user specified weights. Defaults to List[List[int]].

    Returns:
        [type]: [description]
    """
    filter = np.array(weights)
    median = applyMedianFilter(image, filter)
    return median

def exportImage(image: np.array, filename: str) -> None:
    """export image to specified location

    Args:
        image (np.array): image to export
        filename (str): file name to create
    """
    img = Image.fromarray(image)
    img = img.convert("L")
    if not os.path.exists(conf["OUTPUT_DIR"]):
        os.makedirs(conf["OUTPUT_DIR"])
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
    print('Processing.......Results will be displayed after completion.....')
    global conf
    conf = toml.load('./config.toml')

    process_batch(conf)

if __name__ == "__main__":
    main()







       

        
       

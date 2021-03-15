# Overview

* Take a batch of images as input. 
* Run the operations on each image
    * Grey Scaling Salt and Pepper Noise, Gaussian Noise, Linear Filtering, Median Filtering, Histogram Calculations, Histogram Equalization, Image Quantization, and Mean Squared Error). 
* All of these operations output a new image for each image in the batch. 
* A TOML file for configuration of attributes like noise and the weights for filters.

## Usage

```
git clone https://github.com/praveenarallabandi/ImageAnalysis.git
cd ImageAnalysis
pip3 install --user pipenv
python ImageAnalysisPart1.py
```

## Implementation

The project implementation is done using Python. Using Python, we can rapidly develop and integrate each operation. Python's NumPy library, which allows for array operations. 

Certain image array operations are time-consuming, and those scenarios were addressed with optimizing NumPy arrays (using NumPy methods as much as possible) and with numba. Numba is an open-source JIT compiler that translates a subset of Python and NumPy code into fast machine code. Numba has a python function decorator for just-in-time compiling functions to machine code before executing. Using this decorator on functions that use heavy math and looping (i.e., filters and noise) provides significant speed increases with speeds similar to using lower-level compiled languages like C/C++ or Rust. For plotting histograms, Python's `matplotlib,` the relatively standard and robust plotting library, outputs plots to a file with the rest of the exported output images.

## Third Party Libraries 

* numpy - For Array operations
* matplotlib - Plot
* numba - H=JIT for speed exectuion
* toml - Configuration settings
* PIL (Image) - Used only for importing and exporting images

 ## Functions

```python
def calculateHistogram(image: np.array) -> np.array:
```
`calculate_histogram` Generates the histogram, equalized histogram, and quantized image based on the equalized histogram

```python
def image_quantization_mse(image: np.array, imageQuant: np.array, imageName: str) -> float:
```
 `image_quantization_mse` Calculates mean square error for two input images

```python
def convertToSingleColorSpectrum(image: np.array, colorSpectrum: str) -> np.array:
```
`convertToSingleColorSpectrum` Generates the NumPy array for a single color spectrum.

```python
def corruptImageGaussian(image: np.array, strength: int) ->  np.array:
```
`corruptImageGaussian` Generates image with gaussian noise applied

```python
def corruptImageSaltAndPepper(image: np.array, strength: int) -> np.array:
```
`corruptImageSaltAndPepper` Generates image with salt and pepper noise applied

```python
def linearFilter(image, maskSize=9, weights = List[List[int]]) -> np.array:
```
`linearFilter` Receives a kernel or matrix of weights as a two-dimensional input list and applies that kernel to a copy of an image. The filter is then applied in loops through each pixel in the image and multiples the neighboring pixels' values by the kernel weights. The larger the kernel, the larger the pixel's neighborhood that affects the pixel. 
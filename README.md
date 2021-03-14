# Overview

* Take a batch of images as input. 
* Run the operations on each image
    * Grey Scaling Salt and Pepper Noise, Gaussian Noise, Linear Filtering, Median Filtering, Histogram Calculations, Histogram Equalization, Image Quantization, and Mean Squared Error). 
* All of these operations output a new image for each image in the batch. 
* A TOML file for configuration of attributes like noise and the weights for filters.

## Usage

```sh
py ImageAnalysisPart1.py

```
## Execution

```
git clone https://github.com/praveenarallabandi/ImageAnalysis.git
cd ImageAnalysis
pip3 install --user pipenv
python ImageAnalysisPart1.py
```

## Implementation

The project implementation is done using Python. Using Python, we can rapidly develop and integrate each operation. Python's NumPy library, which allows for array operations. 

Certain image array operations are time-consuming, and those scenarios were addressed with optimizing NumPy arrays (using NumPy methods as much as possible) and with numba. Numba is an open-source JIT compiler that translates a subset of Python and NumPy code into fast machine code. Numba has a python function decorator for just-in-time compiling functions to machine code before executing. Using this decorator on functions that use heavy math and looping (i.e., filters and noise) provides significant speed increases with speeds similar to using lower-level compiled languages like C/C++ or Rust. For plotting histograms, Python's `matplotlib,` the relatively standard and robust plotting library, outputs plots to a file with the rest of the exported output images.


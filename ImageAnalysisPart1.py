# imports
import os
import numpy as np  

# resolution for images
ROWS = 768    
COLS =  568
TotalPixels = ROWS * COLS

# Process files in directory as a batch
def process_batch():
    basepath = ('./Cancerouscellsmears2')
    with os.scandir(basepath) as entries:
        for entry in entries:
            if entry.is_file():
                print('Processing Image - ' + entry.name)
                process_image(entry)
    return basepath

# Process the input image
def process_image(entry):
    img = np.fromfile(entry, dtype = np.uint8, count = TotalPixels)
    print("Dimension of the image array: ", img.ndim)
    print("Size of the image array: ", img.size)
    # Conversion from 1D to 2D array
    img.shape = (img.size // COLS, COLS)

    # Noise addition functions that will allow to corrupt each image with Gaussian & SP
    resultGaussian = corruptImage('gaussian', img)
    print('Gaussian Result - ')
    print(resultGaussian)
    resultSP = corruptImage('sp', img)
    print('Salt & Pepper Result - ')
    print(resultSP)
    
    # Converting color images to selected single color spectrum
    print("New dimension of the array:", img.ndim)
    print("----------------------------------------------------")
    print(" The 2D array of the original image is: \n", img)
    print("----------------------------------------------------")
    print("The shape of the original image array is: ", img.shape)
    # Save the output image
    print("... Save the output image")
    img.astype('int8').tofile(entry.name + '.raw')
    print("... File successfully saved")

def corruptImage(noise_typ, image):
   if noise_typ == "gaussian":
      row,col= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
   elif noise_typ == "sp":
      row,col = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      # out[coords] = 1
      out[tuple(coords)]

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
basepath = process_batch()





       

        
       
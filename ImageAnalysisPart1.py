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
    print("New dimension of the array:", img.ndim)
    print("----------------------------------------------------")
    print(" The 2D array of the original image is: \n", img)
    print("----------------------------------------------------")
    print("The shape of the original image array is: ", img.shape)
    # Save the output image
    print("... Save the output image")
    img.astype('int8').tofile('NewImage.raw')
    print("... File successfully saved")
    # Closing the file
    entry.close()

basepath = process_batch()





       

        
       
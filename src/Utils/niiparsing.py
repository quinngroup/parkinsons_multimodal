import nibabel as nib
import numpy as np

def load_nii(path):

    img = nib.load(path)
    data = img.get_fdata()

    shape = data.shape
    
    '''
    print(shape) #(193, 229, 193)
    print(img.get_data_dtype()) #float32
    print(type(data)) #numpy.ndarray
    '''
    return data
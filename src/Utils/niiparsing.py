import nibabel as nib
import numpy as np

def load_nii(path):

	img = nib.load(path)
	data = img.get_fdata()

	shape = data.shape

	assert shape[-1] == 1

	# Removing last dimension from data
	return data.reshape((shape[0], shape[1], shape[2]))
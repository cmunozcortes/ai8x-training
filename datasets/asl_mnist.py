import os
import sys

import torchvision
from torchvision import transforms

import ai8x

def asl_get_dataset(data, load_train=True, load_test=True):

	(data_dir, args) = data
	
	training_data_path = os.path.join(data_dir, "sign_mnist_train")
	test_data_path = os.path.join(data_dir, "sign_mnist_test")

	train_dataset = torchvision.datasets.ImageFolder(
		root=training_data_path,
		#transform=train_transform,
	)
	test_dataset = torchvision.datasets.ImageFolder(
		root=test_data_path,
		#transform=test_transform,
	)
	return train_dataset, test_dataset

datasets = [
	{
		'name': 'asl_mnist',
		'input': (1, 28, 28),
		'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12,
		           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
		'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		 		   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,),
		'loader': asl_get_dataset,
	},
]
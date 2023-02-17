import os
import numpy as np
import matplotlib.pyplot as plt
import pyvips as pv
from skimage.transform import resize
from slidl.slide import Slide

class slide_image:
	'''
	Generate Slide image from tile dictionary
	'''
	def __init__(self, slide_slide, stain, classes, **kwargs):
		if stain == 'he':
			self.ranked_class = 'atypia'
		elif stain == 'p53':
			self.ranked_class = 'aberrant_positive_columnar'
		else:
			self.ranked_class = 'positive'

		self.classes = classes

		self.slidl_slide = slide_slide
		self.im = self.slidl_slide.thumbnail(level=4)
		self.tile_dict = self.slidl_slide.tileDictionary
		x_tiles = self.slidl_slide.numTilesInX
		y_tiles = self.slidl_slide.numTilesInY
		self.target_tile_dict = {}

		self.predictions = {class_name: np.zeros((x_tiles, y_tiles)[::-1]) for class_name in classes}

		self.fig, self.ax = plt.subplots()

	def draw_class(self, target=None, threshold=None):
		if target is not None:
			target_class = [target]
		else:
			target_class = self.classes

		for tile_address, tile_entry in self.tile_dict.items():
			for class_name in target_class:
				# extract target tiles
				if class_name == target:
					if threshold is not None:
						self.predictions[class_name][tile_address[1],tile_address[0]] = tile_entry['classifierInferencePrediction'][class_name] if tile_entry['classifierInferencePrediction'][class_name] > threshold else 0
					else:
						self.predictions[class_name][tile_address[1],tile_address[0]] = tile_entry['classifierInferencePrediction'][class_name]

				else:
					self.predictions[class_name][tile_address[0],tile_address[1]] = tile_entry['classifierInferencePrediction'][class_name]

	def plot_thumbnail(self, case_id=None, target=None):
		title = ''
		if case_id is not None:
			title += case_id
		if target is not None:
			self.ax.imshow(resize(self.im, self.predictions[list(self.predictions.keys())[0]].shape))		
			title += "\n" + target
		else:
			self.ax.imshow(resize(self.im, self.predictions[target].shape))
		self.ax.set_title(title)

	def plot_class(self, target=None):
		if target is not None:
			im = self.ax.imshow(self.predictions[target], cmap='plasma',alpha=0.3, vmin=0, vmax=1.0)
			cax = self.fig.add_axes([self.ax.get_position().x1+0.01,self.ax.get_position().y0,0.02,self.ax.get_position().height])
			self.fig.colorbar(im, cax=cax)
		else:
			print('Target not provided.')
			
	def save(self, out_path, file_name, format='.png'):
		print("Visualizing inference for "+ out_path +"...")
		self.fig.savefig(os.path.join(out_path, file_name + format))
	
	def show(self):
		self.fig.show()

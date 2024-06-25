
import os
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import skimage
import cv2

from itertools import product, starmap

from monai.data import PatchWSIDataset
from monai.utils import convert_to_dst_type
# from monai.data.utils import iter_patch_position
from monai.transforms.transform import Transform
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta

from monai.utils import (
	ensure_tuple_size,
	ensure_tuple_rep
)
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype

from monai.utils.enums import CommonKeys, ProbMapKeys, WSIPatchKeys, TransformBackends
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN

import skimage.color
import skimage.exposure
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

def get_valid_patch_size(image_size: Sequence[int], patch_size: Sequence[int] | int | np.ndarray) -> tuple[int, ...]:
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))

def iter_patch_position(
    image_size: Sequence[int],
    patch_size: Sequence[int] | int | np.ndarray,
    start_pos: Sequence[int] = (),
	end_pos: Sequence[int] = (),
    overlap: Sequence[float] | float | Sequence[int] | int = 0.0,
    padded: bool = False,
):
    """
    Yield successive tuples of upper left corner of patches of size `patch_size` from an array of dimensions `image_size`.
    The iteration starts from position `start_pos` in the array, or starting at the origin if this isn't provided. Each
    patch is chosen in a contiguous grid using a rwo-major ordering.

    Args:
        image_size: dimensions of array to iterate over
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        start_pos: starting position in the array, default is 0 for each dimension
        overlap: the amount of overlap of neighboring patches in each dimension.
            Either a float or list of floats between 0.0 and 1.0 to define relative overlap to patch size, or
            an int or list of ints to define number of pixels for overlap.
            If only one float/int number is given, it will be applied to all dimensions. Defaults to 0.0.
        padded: if the image is padded so the patches can go beyond the borders. Defaults to False.

    Yields:
        Tuples of positions defining the upper left corner of each patch
    """

    # ensure patchSize and startPos are the right length
    ndim = len(image_size)
    patch_size_ = get_valid_patch_size(image_size, patch_size)
    start_pos = ensure_tuple_size(start_pos, ndim)
    overlap = ensure_tuple_rep(overlap, ndim)

    # calculate steps, which depends on the amount of overlap
    if isinstance(overlap[0], float):
        steps = tuple(round(p * (1.0 - o)) for p, o in zip(patch_size_, overlap))
    else:
        steps = tuple(p - o for p, o in zip(patch_size_, overlap))

    # calculate the last starting location (depending on the padding)
    end_pos = end_pos if padded else tuple(s - round(p) + 1 for s, p in zip(end_pos, patch_size_))

    # collect the ranges to step over each dimension
    ranges = starmap(range, zip(start_pos, end_pos, steps))

    # choose patches by applying product to the ranges
    return product(*ranges)

class TissueTect(Transform):
	"""
	Creates a binary mask that defines the foreground based on thresholds in RGB or HSV color space.
	This transform receives an RGB (or grayscale) image where by default it is assumed that the foreground has
	low values (dark) while the background has high values (white). Otherwise, set `invert` argument to `True`.

	Args:
		threshold: an int or a float number that defines the threshold that values less than that are foreground.
			It also can be a callable that receives each dimension of the image and calculate the threshold,
			or a string that defines such callable from `skimage.filter.threshold_...`. For the list of available
			threshold functions, please refer to https://scikit-image.org/docs/stable/api/skimage.filters.html
			Moreover, a dictionary can be passed that defines such thresholds for each channel, like
			{"R": 100, "G": "otsu", "B": skimage.filter.threshold_mean}
		hsv_threshold: similar to threshold but HSV color space ("H", "S", and "V").
			Unlike RBG, in HSV, value greater than `hsv_threshold` are considered foreground.
		hed_threshold: similar to threshold but in the HED color space ("H", "E", and "D") for Heamatoxylin & Eosin + DAB.
		invert: invert the intensity range of the input image, so that the dtype maximum is now the dtype minimum,
			and vice-versa.
		enhance: enhance the contrast of the image by increasing the intensity of the pixels.

	"""

	backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

	def __init__(
		self,
		threshold: dict | Callable | str | float | int = "otsu",
		hsv_threshold: dict | Callable | str | float | int | None = None,
		hed_threshold: dict | Callable | str | float | int | None = None,
		l_threshold: dict | Callable | str | float | int | None = None,
		invert: bool = False,
		enhance: bool = False,
	) -> None:
		self.thresholds: dict[str, Callable | float] = {}
		if threshold is not None:
			if isinstance(threshold, dict):
				for mode, th in threshold.items():
					self._set_threshold(th, mode.upper())
			else:
				self._set_threshold(threshold, "R")
				self._set_threshold(threshold, "G")
				self._set_threshold(threshold, "B")
		if hsv_threshold is not None:
			if isinstance(hsv_threshold, dict):
				for mode, th in hsv_threshold.items():
					self._set_threshold(th, mode.upper())
			else:
				self._set_threshold(hsv_threshold, "H")
				self._set_threshold(hsv_threshold, "S")
				self._set_threshold(hsv_threshold, "V")
		if hed_threshold is not None:
			if isinstance(hed_threshold, dict):
				for mode, th in hed_threshold.items():
					self._set_threshold(th, mode.upper())
			else:
				self._set_threshold(hed_threshold, "H")
				self._set_threshold(hed_threshold, "E")
				self._set_threshold(hed_threshold, "D")
		if l_threshold is not None:
			if isinstance(l_threshold, dict):
				for mode, th in l_threshold.items():
					self._set_threshold(th, mode.upper())
			else:
				self._set_threshold(l_threshold, "L")

		self.thresholds = {k: v for k, v in self.thresholds.items() if v is not None}
		if self.thresholds.keys().isdisjoint(set("RGBHSVLED")):
			raise ValueError(
				f"Threshold for at least one channel of RGB, HSV, HED, or grayscale ('L') needs to be set. {self.thresholds} is provided."
			)
		self.invert = invert
		self.enhance = enhance

	def _set_threshold(self, threshold, mode):
		if callable(threshold):
			self.thresholds[mode] = threshold
		elif isinstance(threshold, str):
			self.thresholds[mode] = getattr(skimage.filters, "threshold_" + threshold.lower())
		elif isinstance(threshold, (float, int)):
			self.thresholds[mode] = float(threshold)
		else:
			raise ValueError(
				f"`threshold` should be either a callable, string, or float number, {type(threshold)} was given."
			)

	def _get_threshold(self, image, mode):
		threshold = self.thresholds.get(mode)
		if callable(threshold):
			return threshold(image)
		return threshold

	def _enhance_contrast(self, image, factor = 2):
		#contast enhancement
		mean = np.uint8(cv2.mean(cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY))[0])
		img_deg = np.ones_like(image.transpose(1, 2, 0)) * mean
		img = cv2.addWeighted(image.transpose(1, 2, 0), factor, img_deg, 1-factor, 0.0)
		return img.transpose(2, 0, 1)
	
	def __call__(self, image: NdarrayOrTensor):
		image = convert_to_tensor(image, track_meta=get_track_meta())
		img_rgb, *_ = convert_data_type(image, np.ndarray)
		if self.invert:
			img_rgb = skimage.util.invert(img_rgb)
		if self.enhance:
			img_rgb = self._enhance_contrast(img_rgb)
		foregrounds = []
		if not self.thresholds.keys().isdisjoint(set("RGB")):
			rgb_foreground = np.zeros_like(img_rgb[:1])
			for img, mode in zip(img_rgb, "RGB"):
				threshold = self._get_threshold(img, mode)
				if threshold:
					rgb_foreground = np.logical_or(rgb_foreground, img <= threshold)
			foregrounds.append(rgb_foreground)
		if not self.thresholds.keys().isdisjoint(set("HSV")):
			img_hsv = skimage.color.rgb2hsv(img_rgb, channel_axis=0)
			hsv_foreground = np.zeros_like(img_rgb[:1])
			for img, mode in zip(img_hsv, "HSV"):
				threshold = self._get_threshold(img, mode)
				if threshold:
					hsv_foreground = np.logical_or(hsv_foreground, img > threshold)
			foregrounds.append(hsv_foreground)
		if not self.thresholds.keys().isdisjoint(set("HED")):
			#transformation to hed space
			img_hed = skimage.color.rgb2hed(img_rgb, channel_axis=0)
			hed_foreground = np.zeros_like(img_rgb[:1])
			for img, mode in zip(img_hed, "HED"):
				threshold = self._get_threshold(img, mode)
				if threshold:
					hed_foreground = np.logical_or(hed_foreground, img != 0)
			foregrounds.append(hed_foreground)

		mask = np.stack(foregrounds).all(axis=0)

		return convert_to_dst_type(src=mask, dst=image)[0]

class MaskedPatchWSIDataset(PatchWSIDataset):
	"""
	This dataset extracts patches from whole slide images at the locations where foreground mask
	at a given level is non-zero.

	Args:
		data: the list of input samples including image, location, and label (see the note below for more details).
		size: the size of patch to be extracted from the whole slide image.
		level: the level at which the patches to be extracted (default to 0).
		mask_level: the resolution level at which the mask is created.
		transform: transforms to be executed on input data.
		include_label: whether to load and include labels in the output
		center_location: whether the input location information is the position of the center of the patch
		additional_meta_keys: the list of keys for items to be copied to the output metadata from the input data
		reader: the module to be used for loading whole slide imaging. Defaults to cuCIM. If `reader` is

			- a string, it defines the backend of `monai.data.WSIReader`.
			- a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader,
			- an instance of a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

		kwargs: additional arguments to pass to `WSIReader` or provided whole slide reader class

	Note:
		The input data has the following form as an example:

		.. code-block:: python

			[
				{"image": "path/to/image1.tiff"},
				{"image": "path/to/image2.tiff", "mask": "path/to/mask2.tiff"}
				{"image": "path/to/image3.tiff", "size": [20, 20], "level": 2}
			]

	"""

	def __init__(
		self,
		data: Sequence,
		patch_size: Optional[Union[int, Tuple[int, int]]] = None,
		patch_level: Optional[int] = None,
		mask_level: int = 7,
		mask_fill: bool = True,
		save_mask: bool = False,
		transform: Optional[Callable] = None,
		include_label: bool = False,
		center_location: bool = False,
		additional_meta_keys: Sequence[str] = (ProbMapKeys.LOCATION, ProbMapKeys.NAME),
		reader="cuCIM",
		**kwargs,
	):
		super().__init__(
			data=[],
			patch_size=patch_size,
			patch_level=patch_level,
			transform=transform,
			include_label=include_label,
			center_location=center_location,
			additional_meta_keys=additional_meta_keys,
			reader=reader,
			**kwargs,
		)

		self.mask_level = mask_level
		self.mask_fill = mask_fill
		self.save_mask = save_mask
		# Create single sample for each patch (in a sliding window manner)
		self.data: list
		self.image_data = list(data)
		for sample in self.image_data:
			patch_samples = self._evaluate_patch_locations(sample)
			self.data.extend(patch_samples)

	def _fill_holes(self, mask, min_pixel_count=30):
		contours_filled_mask, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		polygons = []
		for contour in contours_filled_mask:
			if len(contour) >= min_pixel_count:
				polygon = Polygon(contour.reshape(-1, 2))
				polygons.append(polygon)
		buffered_polygons = [polygon.buffer(3) for polygon in polygons]
		merged_polygons = unary_union(buffered_polygons)
		if merged_polygons.geom_type == 'Polygon':
			merged_polygons = [merged_polygons]
		if isinstance(merged_polygons, MultiPolygon):
			merged_polygons = list(merged_polygons.geoms)
		if isinstance(merged_polygons, GeometryCollection):
			merged_polygons = list(merged_polygons.geoms)

		# Create a blank image with the same shape as the original image
		filled_mask = np.zeros_like(mask)

		# Loop over each polygon
		for polygon in polygons:
			# Get the x and y coordinates of the polygon
			x, y = polygon.exterior.coords.xy
			# Convert the polygon coordinates to integer
			poly_coords = np.array([list(zip(x, y))], dtype=np.int32)
			# Fill the polygon area in the mask with 1
			cv2.fillPoly(filled_mask, poly_coords, 1)

		return filled_mask

	def _evaluate_patch_locations(self, sample):
		"""Calculate the location for each patch based on the mask at different resolution level"""
		patch_size = self._get_size(sample)
		patch_level = self._get_level(sample)
		wsi_obj = self._get_wsi_object(sample)

		# load the entire image at level=mask_level
		wsi, _ = self.wsi_reader.get_data(wsi_obj, level=self.mask_level)

		# create the foreground tissue mask and get all indices for non-zero pixels
		mask = np.squeeze(convert_to_dst_type(TissueTect(hed_threshold={'E': 1}, enhance=True)(wsi), dst=wsi)[0])
		if self.mask_fill:
			mask = self._fill_holes(mask)
		if self.save_mask:
			cv2.imwrite(sample['mask'], mask * 255)
		mask_locations = np.vstack(mask.nonzero()).T

		# convert mask locations to image locations at level=0
		mask_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, self.mask_level)
		patch_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, patch_level)
		patch_size_0 = np.array([p * patch_ratio for p in patch_size])  # patch size at level 0
		patch_locations = np.round((mask_locations + 0.5) * float(mask_ratio) - patch_size_0 // 2).astype(int)

		# fill out samples with location and metadata
		sample[WSIPatchKeys.SIZE.value] = patch_size
		sample[WSIPatchKeys.LEVEL.value] = patch_level
		sample[ProbMapKeys.NAME.value] = os.path.basename(sample[CommonKeys.IMAGE])
		sample[ProbMapKeys.COUNT.value] = len(patch_locations)
		sample[ProbMapKeys.SIZE.value] = mask.shape
		import ipdb; ipdb.set_trace()
		return [
			{**sample, WSIPatchKeys.LOCATION.value: np.array(loc), ProbMapKeys.LOCATION.value: mask_loc}
			for loc, mask_loc in zip(patch_locations, mask_locations)
		]

class SlidingPatchWSIDataset(PatchWSIDataset):
	"""
	This dataset extracts patches from whole slide images (without loading the whole image)
	It also reads labels for each patch and provides each patch with its associated class labels.

	Args:
		data: the list of input samples including image, location, and label (see the note below for more details).
		size: the size of patch to be extracted from the whole slide image.
		level: the level at which the patches to be extracted (default to 0).
		offset: the offset of image to extract patches (the starting position of the upper left patch).
		offset_limits: if offset is set to "random", a tuple of integers defining the lower and upper limit of the
			random offset for all dimensions, or a tuple of tuples that defines the limits for each dimension.
		overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
			If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
		transform: transforms to be executed on input data.
		reader: the module to be used for loading whole slide imaging. Defaults to cuCIM. If `reader` is

			- a string, it defines the backend of `monai.data.WSIReader`.
			- a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader,
			- an instance of a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

		map_level: the resolution level at which the output map is created.
		seed: random seed to randomly generate offsets. Defaults to 0.
		kwargs: additional arguments to pass to `WSIReader` or provided whole slide reader class

	Note:
		The input data has the following form as an example:

		.. code-block:: python

			[
				{"image": "path/to/image1.tiff"},
				{"image": "path/to/image2.tiff", "patch_size": [20, 20], "patch_level": 2}
				{"image": "path/to/image3.tiff", "mask": "path/to/mask3.tiff"}
			]

	"""

	def __init__(
		self,
		data: Sequence,
		patch_size: Optional[Union[int, Tuple[int, int]]] = None,
		patch_level: Optional[int] = None,
		mask_level: int = 7,
		mask_fill: bool = True,
		save_mask: bool = False,
		overlap: Union[Tuple[float, float], float] = 0.0,
		offset: Union[Tuple[int, int], int, str] = (0, 0),
		offset_limits: Optional[Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int]]] = None,
		transform: Optional[Callable] = None,
		include_label: bool = False,
		center_location: bool = False,
		additional_meta_keys: Sequence[str] = (ProbMapKeys.LOCATION, ProbMapKeys.SIZE, ProbMapKeys.COUNT),
		reader="cuCIM",
		**kwargs,
	):
		super().__init__(
			data=[],
			patch_size=patch_size,
			patch_level=patch_level,
			transform=transform,
			include_label=include_label,
			center_location=center_location,
			additional_meta_keys=additional_meta_keys,
			reader=reader,
			**kwargs,
		)
		
		self.overlap = overlap
		self.random_offset = False
		# Set the mask config
		self.mask_level = mask_level
		self.mask_fill = mask_fill
		self.save_mask = save_mask
		# Create single sample for each patch (in a sliding window manner)
		self.data: list
		self.image_data = list(data)
		for sample in self.image_data:
			patch_samples = self._evaluate_patch_locations(sample)
			self.data.extend(patch_samples)

	def _fill_holes(self, mask, min_pixel_count=30):
		contours_filled_mask, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		polygons = []
		for contour in contours_filled_mask:
			if len(contour) >= min_pixel_count:
				polygon = Polygon(contour.reshape(-1, 2))
				polygons.append(polygon)
		buffered_polygons = [polygon.buffer(3) for polygon in polygons]
		merged_polygons = unary_union(buffered_polygons)
		if merged_polygons.geom_type == 'Polygon':
			merged_polygons = [merged_polygons]
		if isinstance(merged_polygons, MultiPolygon):
			merged_polygons = list(merged_polygons.geoms)
		if isinstance(merged_polygons, GeometryCollection):
			merged_polygons = list(merged_polygons.geoms)

		# Create a blank image with the same shape as the original image
		filled_mask = np.zeros_like(mask)

		# Loop over each polygon
		for polygon in polygons:
			# Get the x and y coordinates of the polygon
			x, y = polygon.exterior.coords.xy
			# Convert the polygon coordinates to integer
			poly_coords = np.array([list(zip(x, y))], dtype=np.int32)
			# Fill the polygon area in the mask with 1
			cv2.fillPoly(filled_mask, poly_coords, 1)

		return filled_mask

	def _get_offset(self, sample):
		if self.random_offset:
			if self.offset_limits is None:
				offset_limits = tuple((-s, s) for s in self._get_size(sample))
			else:
				offset_limits = self.offset_limits
			return tuple(self.R.randint(low, high) for low, high in offset_limits)
		return self.offset

	def get_bboxes(self, mask, max_dist=250, filter=True, max_bboxes=2):
		labeled_mask = label(mask)

		# get bounding box of the mask
		props = sorted(regionprops(labeled_mask), key=lambda x: x.area, reverse=True)

		# Get the centroids of the bounding boxes
		centroids = [prop.centroid for prop in props]

		# Apply DBSCAN clustering
		eps = max_dist  # maximum distance between two samples for them to be considered as in the same neighborhood
		min_samples = 1  # number of samples in a neighborhood for a point to be considered as a core point
		clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
		labels = clustering.labels_

		# Combine bounding boxes that belong to the same cluster
		combined_props, bounding_masks, full_masks = [], [], []
		for cluster_id in np.unique(labels):
			if cluster_id != -1:  # ignore noise (cluster_id = -1)
				cluster_props = [props[i] for i in np.where(labels == cluster_id)[0]]
				minr = min(prop.bbox[0] for prop in cluster_props)
				minc = min(prop.bbox[1] for prop in cluster_props)
				maxr = max(prop.bbox[2] for prop in cluster_props)
				maxc = max(prop.bbox[3] for prop in cluster_props)
				combined_props.append((minr, minc, maxr, maxc))
				# Create a single mask for the entire cluster
				bounding_mask = np.zeros((maxr-minr, maxc-minc))
				full_mask = np.zeros_like(mask)
				for prop in cluster_props:
					full_mask[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] = prop.filled_image
					bounding_mask[prop.bbox[0]-minr:prop.bbox[2]-minr, prop.bbox[1]-minc:prop.bbox[3]-minc] = prop.filled_image
				bounding_masks.append(bounding_mask)
				full_masks.append(full_mask)

		if filter:
			# Order masks by area
			areas = [np.sum(i) for i in bounding_masks]
			sorted_indices = np.argsort(areas)[::-1]

			bounding_masks = [bounding_masks[i] for i in sorted_indices]
			self.mask = np.maximum.reduce([full_masks[i] for i in sorted_indices][:max_bboxes])
		
		self.bboxes = combined_props
		return self.bboxes

	def _evaluate_patch_locations(self, sample):
		"""Calculate the location for each patch in a sliding-window manner"""
		patch_size = self._get_size(sample)
		patch_level = self._get_level(sample)
		wsi_obj = self._get_wsi_object(sample)

		# load the entire image at level=mask_level
		wsi, _ = self.wsi_reader.get_data(wsi_obj, level=self.mask_level)

		# create the foreground tissue mask and get all indices for non-zero pixels
		mask = np.squeeze(convert_to_dst_type(TissueTect(hed_threshold={'E': 1}, enhance=True)(wsi), dst=wsi)[0])
		if self.mask_fill:
			mask = self._fill_holes(mask)
		if self.save_mask:
			cv2.imwrite(sample['mask'], mask * 255)
		mask_locations = np.vstack(mask.nonzero()).T

		# calculate the locations
		wsi_size = self.wsi_reader.get_size(wsi_obj, patch_level)
		mask_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, self.mask_level)
		patch_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, patch_level)
		patch_size_0 = np.array([p * patch_ratio for p in patch_size])  # patch size at level 0
		# offset = self._get_offset(sample)
		
		bboxes = self.get_bboxes(mask)
		downsized_bboxes = []
		for bbox in bboxes:
			downsized_bboxes.append(tuple([int(mask_ratio*x) for x in bbox]))
		
		total_patch_locations = []
		# Create tiles
		for bbox in downsized_bboxes:
			patch_locations = list(
					iter_patch_position(
						image_size=wsi_size, patch_size=patch_size_0, start_pos=((bbox[0], bbox[1])), end_pos=(bbox[2], bbox[3]), overlap=self.overlap, padded=False
					)
				)
			total_patch_locations.extend(patch_locations)
		patch_locations = np.array(total_patch_locations)
		
		# convert locations to mask_location
		mask_locations = np.round((patch_locations + patch_size_0 // 2) / float(mask_ratio))

		# fill out samples with location and metadata
		sample[WSIPatchKeys.SIZE.value] = patch_size
		sample[WSIPatchKeys.LEVEL.value] = patch_level
		sample[ProbMapKeys.NAME.value] = os.path.basename(sample[CommonKeys.IMAGE])
		sample[ProbMapKeys.COUNT.value] = len(patch_locations)
		sample[ProbMapKeys.SIZE.value] = mask.shape
		return [
            {**sample, WSIPatchKeys.LOCATION.value: np.array(loc), ProbMapKeys.LOCATION.value: mask_loc}
            for loc, mask_loc in zip(patch_locations, mask_locations)
        ]

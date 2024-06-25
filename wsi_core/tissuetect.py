import os
import openslide
import numpy as np
from PIL import ImageEnhance
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from shapely.affinity import scale
from shapely.ops import unary_union
from skimage.util import invert
from skimage.morphology import dilation, square
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN
from itertools import starmap
import cv2
from skimage.color import rgb2hed, hed2rgb, rgb2hsv, rgb2gray

class TissueDetector:
    def __init__(self, wsi_path, extensions = ('.svs', '.ndpi','.tiff','.tif'), invert = False, resize=False, artifact_detection=True,
                 mask_level=None, min_mask_width=1200, pen_markers=False, blue_marker=False, red_marker=False, black_marker=False, green_marker=False):
        print('Tissue Segmentation')

        self.wsi_path = wsi_path
        self.extensions = extensions
        if self.wsi_path is not None:
            if os.path.isfile(wsi_path) and wsi_path.endswith(extensions):
                print('Reading file: ', wsi_path)
            else:
                raise Exception(f'No files found in {wsi_path} with extensions {extensions}')
        else:
            raise('Please specify either a directory or a single file')

        #TODO: load mask if specifed rather than WSI
        self.wsi_obj = openslide.OpenSlide(wsi_path)
        print(f'Loaded Whole Slide Image: {self.wsi_path}')

        if mask_level < 0:
            self.mask_level = self.get_best_mask_level(min_mask_width)
        else:
            self.mask_level = mask_level

        self.wsi_level = self.wsi_obj.read_region((0, 0), self.mask_level, self.wsi_obj.level_dimensions[self.mask_level])
        if resize:
            aspect_ratio = self.wsi_level.size[1] / self.wsi_level.size[0]
            new_height = int(min_mask_width * aspect_ratio)
            self.wsi_level = self.wsi_level.resize((min_mask_width, new_height))
            print('resized to: ', self.wsi_level.size)

        self.mask = np.zeros_like(self.wsi_level)
        self.bboxes = None
        self.contours = None
        self.polygons = None

        self.artifact_detection = artifact_detection
        self.pen_markers = pen_markers
        if self.artifact_detection or self.pen_markers:
            self.artifact_mask = np.zeros_like(self.mask)
            if pen_markers:
                blue_marker = black_marker = red_marker = green_marker = True
            self.blue_marker = blue_marker
            self.red_marker = red_marker
            self.black_marker = black_marker
            self.green_marker = green_marker
        
        self.invert = invert

    def _transpose(img):
        return img.T
    
    def _invert(img):
        return invert(img)
    
    def get_level_dimensions(self, level):
        return self.wsi_obj.level_dimensions[level]

    def _enhance_contrast(image, factor = 1):
        #contast enhancement
        mean = np.uint8(cv2.mean(cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY))[0])
        img_deg = np.ones_like(image.transpose(1, 2, 0)) * mean
        img = cv2.addWeighted(image.transpose(1, 2, 0), factor, img_deg, 1-factor, 0.0)
        return img.transpose(2, 0, 1)

    def get_best_mask_level(self, min_mask_width=1200):
        level = self.wsi_obj.level_count - 1
        while self.wsi_obj.level_dimensions[level][0] < min_mask_width:
            print('level dimensions: ', self.wsi_obj.level_dimensions[level], 'for level: ', level)
            level -= 1
        print('level picked: ', level, 'with dimensions: ', self.wsi_obj.level_dimensions[level])
        return level

    def mask_to_polygons(self, mask=None, min_pixel_count=30):
        # Create a blank image with the same shape as the original image
        mask = mask if mask is not None else self.mask
        self.contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in self.contours:
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
        print('Number of polygons: ', len(merged_polygons))
        return merged_polygons

    def mask_contours(self, mask=None):
        mask = mask if mask is not None else self.mask
        self.contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self.contours

    def mask_bboxes(self, max_dist=250, max_bboxes=2, filter=True):
        labeled_mask = label(self.mask)

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
                full_mask = np.zeros_like(self.mask)
                for prop in cluster_props:
                    full_mask[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] = prop.filled_image
                    bounding_mask[prop.bbox[0]-minr:prop.bbox[2]-minr, prop.bbox[1]-minc:prop.bbox[3]-minc] = prop.filled_image
                bounding_masks.append(bounding_mask)
                full_masks.append(full_mask)
 
        self.bboxes = combined_props

        if filter:
            # Order masks by area
            areas = [np.sum(i) for i in bounding_masks]
            sorted_indices = np.argsort(areas)[::-1]

            bounding_masks = [bounding_masks[i] for i in sorted_indices]
            self.mask = np.maximum.reduce([full_masks[i] for i in sorted_indices][:max_bboxes])
        
            self.bboxes = self.bboxes[:max_bboxes]
        
        return self.bboxes

    def create_patches(self, bboxes, patch_size, patch_level, overlap):
        bboxes if bboxes else self.bboxes
        level_dim = self.get_level_dimensions(patch_level)
        x_factor = level_dim[0]/self.mask.shape[1]
        y_factor = level_dim[1]/self.mask.shape[0]

        steps = patch_size - patch_size * overlap

        #Scale bboxes
        scaled_bboxes = []
        for bbox in bboxes:
            x_min, x_max = int(x_factor*bbox[0]), int(x_factor*bbox[1])
            y_min, y_max = int(y_factor*bbox[2]), int(y_factor*bbox[3])
            scaled_bboxes.append(tuple([x_min, x_max, y_min, y_max]))

        #Scale Polygons

        for bbox in scaled_bboxes:
            patches = starmap(range, zip((bbox[0], bbox[2]), (bbox[1], bbox[3]), (steps, steps)))
            for patch in patches:
                yield patch

    def filter_tiles(self, tiles, patch_level, polygons, mode='center'):
        polygons if polygons else self.polygons
        level_dim = self.get_level_dimensions(patch_level)
        x_factor = level_dim[0]/self.mask.shape[1]
        y_factor = level_dim[1]/self.mask.shape[0]

        scaled_polygons = scale(MultiPolygon(polygons), xfact=x_factor, yfact=y_factor)
        #Check if center of bbox is in Polygon
        for patch in tiles:
            if mode == 'center': # check if center of bbox is in Polygon
                if scaled_polygons.contains(Point(patch[2]//2, patch[3]//2)):
                    yield patch
            elif mode == 'basic': #Check if top left corner of bbox is in Polygon
                if scaled_polygons.contains(Point(patch[0], patch[1])):
                    yield patch
            elif mode == 'any': #Check if any point of bbox is in Polygon
                if scaled_polygons.contains(Point(patch[0], patch[1])):
                    yield patch
                elif scaled_polygons.contains(Point(patch[0], patch[3])):
                    yield patch
                elif scaled_polygons.contains(Point(patch[2], patch[1])):
                    yield patch
                elif scaled_polygons.contains(Point(patch[2], patch[3])):
                    yield patch
                else:
                    continue
            elif mode == 'all': #Check if all points of bbox are in Polygon
                if scaled_polygons.contains(Point(patch[0], patch[1])) and scaled_polygons.contains(Point(patch[0], patch[3])) and scaled_polygons.contains(Point(patch[2], patch[1])) and scaled_polygons.contains(Point(patch[2], patch[3])):
                    yield patch
                else:
                    continue
            else:
                raise ValueError('Invalid mode. Choose from: center, basic, any')

    def get_hed_mask(self, hed_contrast=1):
        wsi_rgb = self.wsi_level.convert('RGB')
        enhancer = ImageEnhance.Contrast(wsi_rgb)
        # increase contrast
        wsi_rgb = enhancer.enhance(hed_contrast)
        # Separate the stains from the IHC image
        img_hed = rgb2hed(wsi_rgb)
        # Create an RGB image for each of the stains
        foreground = np.zeros_like(img_hed[:, :, 0])
        ihc = hed2rgb(np.stack((img_hed[:, :, 1], foreground, foreground), axis=-1)) # we are only intrested in 'e' of 'hed'
        contrast_mask = ihc.copy()
        contrast_mask[contrast_mask==1] = 0
        contrast_mask[contrast_mask!=0] = 1
        contrast_mask = rgb2gray(contrast_mask)
        
        # Count the number of 1s in the mask
        count_ones = np.count_nonzero(contrast_mask)
        # Get the total number of elements in the mask
        total_elements = contrast_mask.size
        # Calculate the ratio of 1s
        coverage = np.round(count_ones / total_elements * 100)
        #print(count_ones, total_elements, coverage)
        return contrast_mask, coverage

    def get_gray_mask(self, gray_contrast=20, gray_threshold=210):
        wsi_level_gray = self.wsi_level.convert('L')
        enhancer = ImageEnhance.Contrast(wsi_level_gray)
        # increase contrast
        wsi_contrast = enhancer.enhance(gray_contrast)
        # filter gray
        gray_mask = np.array(wsi_contrast) > gray_threshold
        # remove spots where gray_mask is 255 and replace with 0
        gray_mask = gray_mask.astype(np.uint8)
        gray_mask[gray_mask == 0] = 255
        contrast_mask = 1-gray_mask
        count_ones = np.count_nonzero(contrast_mask)
        # Get the total number of elements in the mask
        total_elements = contrast_mask.size
        # Calculate the ratio of 1s
        coverage = np.round(count_ones / total_elements * 100)
        return contrast_mask, coverage

    def pen_marker_mask(self, min_width=1200, kernel=(5, 5), black_marker=True, blue_marker=True, green_marker=True, red_marker=True, dilute=False):
        self.artifact_mask = np.zeros_like(self.mask)
        wsi_level_hsv = rgb2hsv(np.array(self.wsi_level)[:,:,:3])
        wsi_level_hsv = (wsi_level_hsv * 255).astype('uint8')
        if black_marker:
            black = cv2.inRange(wsi_level_hsv, np.array([0, 0, 0]).astype('uint8'), np.array([255, 255, 165]).astype('uint8'))
            # filter black mask
            black_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
            black = cv2.morphologyEx(black, cv2.MORPH_OPEN, black_kernel)
            _, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.artifact_mask = cv2.bitwise_or(self.artifact_mask.astype('uint8'), black.astype('uint8'))
        if blue_marker:
            blue = cv2.inRange(wsi_level_hsv, np.array([130, 50, 30]).astype('uint8'), np.array([180,255,255]).astype('uint8'))
            self.artifact_mask = cv2.bitwise_or(self.artifact_mask.astype('uint8'), blue.astype('uint8'))
        if green_marker:
            green = cv2.inRange(wsi_level_hsv, np.array([30, 30, 50]).astype('uint8'), np.array([130, 255, 255]).astype('uint8'))
            self.artifact_mask = cv2.bitwise_or(self.artifact_mask.astype('uint8'), green.astype('uint8'))
        if red_marker:
            red1 = cv2.inRange(wsi_level_hsv, np.array([0, 30, 30]).astype('uint8'), np.array([30, 255, 255]).astype('uint8'))
            red2 = cv2.inRange(wsi_level_hsv, np.array([200, 100, 100]).astype('uint8'), np.array([255, 255, 255]).astype('uint8'))
            red = cv2.bitwise_or(red1,red2)
            # filter red mask
            red_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
            red = cv2.morphologyEx(red, cv2.MORPH_OPEN, red_kernel)
            _, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.artifact_mask = cv2.bitwise_or(self.artifact_mask.astype('uint8'), red.astype('uint8'))

        # dilute pen markers
        if dilute:
            selem = square(int(min_width/400)) #TODO
            self.artifact_mask = dilation(self.artifact_mask, selem)

        return self.artifact_mask

    def get_mask(self, cs='hed', min_width=1200, min_pixel_count=30, kernel=(5,5), dilute=False, he_cutoff_percent=5, artifact_detection=True):
        # Automatically detect best colour space for tissue detection
        if cs == 'hed':
            self.mask, coverage = self.get_hed_mask()
        elif (cs == 'gray') or (coverage < he_cutoff_percent) or (coverage == 100.0):
            self.mask, _ = self.get_gray_mask()
        else:
            NotImplementedError('Only HED and Gray tissue detection is currently supported')

        self.artifact_detection = artifact_detection
        if self.artifact_detection:
            tissue_mask_normalized = cv2.normalize(self.mask, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            pen_markers = self.pen_marker_mask(min_width, kernel, dilute=dilute)
            self.mask = np.bitwise_and(tissue_mask_normalized.astype('int'), np.bitwise_not(pen_markers.astype('int')))
        filled_mask = self.mask.astype(np.uint8)
        
        polygon_mask = np.zeros_like(filled_mask, dtype=np.uint8)
        self.polygons = self.mask_to_polygons(filled_mask, min_pixel_count)

        # Loop over each polygon
        for polygon in self.polygons:
            # Get the x and y coordinates of the polygon
            x, y = polygon.exterior.coords.xy
            # Convert the polygon coordinates to integer
            poly_coords = np.array([list(zip(x, y))], dtype=np.int32)
            # Fill the polygon area in the mask with 1
            cv2.fillPoly(polygon_mask, poly_coords, 1)
    
        self.mask = polygon_mask
        self.contours = self.mask_contours()

        if self.invert:
            self.mask = self._invert(self.mask)

        return self.mask

    def save_mask(self, save_file, suffix='_mask.png',):
        if save_file is None:
            save_path = self.wsi_path
        else:
            save_path = os.path.dirname(os.path.abspath(save_file))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print('Created directory: ', save_path)
        save_file = os.path.join(save_path, '.'.join(os.path.basename(self.wsi_path).split('.')[:-1])+suffix)

        # Scale mask values to 8-bit range
        mask = (self.mask * 255).astype(np.uint8)
        cv2.imwrite(save_file, mask)
        print('Saved mask to: ', save_path)

    def save_contours(self, save_file, suffix='_contours.png', colour=(255,0,0), line_thickness=3):
        if save_file is None:
            save_path = self.wsi_path
        else:
            save_path = os.path.dirname(os.path.abspath(save_file))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print('Created directory: ', save_path)

        wsi_rgb = np.array(self.wsi_level.convert('RGB'))
        contours = self.mask_contours()
        cv2.drawContours(wsi_rgb, contours, -1, colour, line_thickness, lineType=cv2.LINE_8)
        save_file = os.path.join(save_path, '.'.join(os.path.basename(self.wsi_path).split('.')[:-1])+suffix)

        cv2.imwrite(save_file, wsi_rgb)

    def save_bboxes(self):
        #TODO
        return

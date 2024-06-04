from openslide import OpenSlide
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches


def get_args():
    parser = argparse.ArgumentParser(description="This script removes the positive control from the mask by only keeping the two biggest connected components")
    parser.add_argument("--mask_path", help="directory of the masks")
    parser.add_argument("--mask_ending", default="_tissuetector.png", help="ending of the mask files")
    parser.add_argument("--save_boundingbox", default=False, help="save the bounding box of the positive control")
    parser.add_argument("--save_path", default=None, help="directory to save the masks; default is same as input dir")
    parser.add_argument("--save_ending", default="_nocontrol.png", help="ending of the saved mask files")
    parser.add_argument("--save_boundingbox_ending", default="_boundingbox.png", help="ending of the saved bounding box files")
    parser.add_argument("--num_tissues", default=2, help="number of tissues to keep")
    parser.add_argument("--max_dist", default=250, type=float, help="maximum distance between two samples for them to be considered as in the same neighborhood")
    return parser.parse_args()

def remove_positive_control(mask_file, save_mask_filename, max_dist=250, save_boundingbox_filename=None,
                            num_tissues=2):
    mask = plt.imread(mask_file).astype(np.int8)

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

    # Order masks by area
    areas = [np.sum(i) for i in bounding_masks]
    sorted_indices = np.argsort(areas)[::-1]
    # Only pick two biggest ones as in DELTA and BEST2 we always have two tissues
    bounding_masks = [bounding_masks[i] for i in sorted_indices]
    full_mask = np.maximum.reduce([full_masks[i] for i in sorted_indices][:num_tissues])
            
    if save_boundingbox_filename is not None:
        # Plot the combined bounding boxes
        plt.figure(figsize=(10,10))
        plt.imshow(mask)
        for minr, minc, maxr, maxc in combined_props:
            rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
        plt.contour(mask, colors='k', linewidths=1)
        plt.axis('off')
        #plt.show()
        plt.savefig(save_boundingbox_filename, bbox_inches='tight', pad_inches=0)
        print(f"Saved mask to {save_boundingbox_filename}")

    # Plot the tissue masks
    plt.figure(figsize=(10,10))
    plt.imshow(full_mask, cmap='gray')
    plt.axis('off')
    #plt.show()
    plt.savefig(save_mask_filename, bbox_inches='tight', pad_inches=0)
    print(f"Saved mask to {save_mask_filename}")

    return bounding_masks, full_mask

if __name__ == '__main__':
    args = get_args()
    mask_path = args.mask_path
    mask_ending = args.mask_ending
    save_path = args.save_path
    save_ending = args.save_ending
    save_boundingbox = args.save_boundingbox
    save_boundingbox_ending = args.save_boundingbox_ending
    num_tissues = args.num_tissues
    max_dist = args.max_dist

    if save_path is None:
        save_path = mask_path

    mask_files = [f for f in os.listdir(mask_path) if f.endswith(mask_ending)]

    for mask_file in mask_files:
        print(f"Processing {mask_file}")
        save_mask = os.path.join(save_path, mask_file.replace(mask_ending, save_ending))
        save_boundingbox = os.path.join(save_path, mask_file.replace(mask_ending, save_boundingbox_ending))
        bounding_masks, full_mask = remove_positive_control(os.path.join(mask_path, mask_file), save_mask, max_dist, save_boundingbox, num_tissues)

    print("Done!")
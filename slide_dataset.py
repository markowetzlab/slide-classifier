import os
import numpy as np
from typing import Callable, Sequence, Tuple, Any
import torchvision.transforms.functional as F
from monai.data import PatchWSIDataset
from monai.transforms import apply_transform
from monai.utils import convert_to_dst_type
from monai.utils.enums import CommonKeys, ProbMapKeys, WSIPatchKeys

class WSIDataset(PatchWSIDataset):
    def __init__(self, data: Sequence, patch_size: int | Tuple[int, int] | None = None, patch_level: int | None = None, transform: Callable[..., Any] | None = None, include_label: bool = True, center_location: bool = True, additional_meta_keys: Sequence[str] | None = None, reader="cuCIM", resize=False, **kwargs):
        super().__init__(data, patch_size, patch_level, transform, include_label, center_location, additional_meta_keys, reader, **kwargs)
        self.resize = resize

    def _transform(self, index: int):
        # Get a single entry of data
        sample: dict = self.data[index]

        # Extract patch image and associated metadata
        image, metadata = self._get_data(sample)

        # Add additional metadata from sample
        for key in self.additional_meta_keys:
            metadata[key] = sample[key]

        # Create output for image
        image = F.to_pil_image(image.transpose(1,2,0))

        if self.transform:
            image = apply_transform(self.transform, image)

        if self.include_label:
            output = {CommonKeys.IMAGE : image}
            output[CommonKeys.LABEL] = self._get_label(sample)
        else:
            output = image

        # Apply transforms and return it
        return output

class MaskedWSIDataset(WSIDataset):
    def __init__(
            self, 
            data: Sequence, 
            patch_size: int | Tuple[int, int] | None = None, 
            patch_level: int | None = None, 
            transform: Callable[..., Any] | None = None, 
            include_label: bool = True, 
            center_location: bool = True, 
            additional_meta_keys: Sequence[str] | None = None, 
            reader="cuCIM", 
            resize=False, 
            **kwargs
        ):
        super().__init__(
            data, 
            patch_size, 
            patch_level, 
            transform, 
            include_label, 
            center_location, 
            additional_meta_keys, 
            reader, 
            **kwargs)
        
        self.resize = resize

    def _evaluate_patch_locations(self, sample):
        """Calculate the location for each patch based on the mask at different resolution level"""
        patch_size = self._get_size(sample)
        patch_level = self._get_level(sample)
        wsi_obj = self._get_wsi_object(sample)

        # load the entire image at level=mask_level
        wsi, _ = self.wsi_reader.get_data(wsi_obj, level=self.mask_level)

        # create the foreground tissue mask and get all indices for non-zero pixels
        mask = np.squeeze(convert_to_dst_type(self._get_mask(hsv_threshold={"S": "otsu"})(wsi), dst=wsi)[0])
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
        return [
            {**sample, WSIPatchKeys.LOCATION.value: np.array(loc), ProbMapKeys.LOCATION.value: mask_loc}
            for loc, mask_loc in zip(patch_locations, mask_locations)
        ]
    
    def _get_mask()

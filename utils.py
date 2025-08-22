import hashlib
from PIL import Image
import numpy as np
import random
import re
import torch

import fiftyone as fo
from fiftyone.operators import types

import cv2
import os
from pathlib import Path

from typing import Callable

def _create_hash():
    """Generate a random hash string.

    Creates a 10-character hash by generating a random integer and hashing it with SHA-256.
    Used for creating unique filenames for transformed images.

    Args:
        None

    Returns:
        str: A 10-character hash string
    """
    randint = random.randint(0, 100000000)  # Generate random integer
    hash = hashlib.sha256(str(randint).encode("utf-8")).hexdigest()[:10]  # Create hash and take first 10 chars
    return hash

def _get_label_fields(sample):
    """Get the names of the fields containing labels for the given sample.
    
    Finds all fields in the sample that contain FiftyOne Label objects.

    Args:
        sample: A FiftyOne sample

    Returns:
        list: Names of fields containing labels
    """
    return [
        field_name
        for field_name in sample.field_names
        if isinstance(sample[field_name], fo.Label)
    ]

def _get_detections_fields(sample, label_fields):
    """Get the names of the fields containing detections for the given sample.
    
    Filters label fields to only those containing Detections objects.

    Args:
        sample: A FiftyOne sample
        label_fields: List of field names to check

    Returns:
        list: Names of fields containing detections
    """
    return [
        field_name
        for field_name in label_fields
        if isinstance(sample[field_name], fo.Detections)
    ]

def _get_keypoints_fields(sample, label_fields):
    """Get the names of the fields containing keypoints for the given sample.
    
    Filters label fields to only those containing Keypoints objects.

    Args:
        sample: A FiftyOne sample
        label_fields: List of field names to check

    Returns:
        list: Names of fields containing keypoints
    """
    return [
        field_name
        for field_name in label_fields
        if isinstance(sample[field_name], fo.Keypoints)
    ]

def _collect_bboxes(sample, label_fields):
    """Collect all bounding boxes and associated info from the given sample.

    Args:
        sample: A FiftyOne sample
        label_fields: List of field names containing bounding boxes

    Returns:
        list: List of [bbox, task_description, detection_id] for each detection
    """
    boxes_list = []
    for field in label_fields:
        detections = sample[field].detections  # Get detections from field
        for det in detections:
            bbox = det.bounding_box
            task_description = det.task_description
            det_id = det.id
            det_info = [bbox, task_description, det_id]
            boxes_list.append(det_info)

    return boxes_list

def _collect_keypoints(sample, keypoints_fields):
    """Collect all keypoints and associated info from the given sample.
    
    Args:
        sample: A FiftyOne sample
        keypoints_fields: List of field names containing keypoints

    Returns:
        list: List of [points, task_description, point_id] for each keypoint set
    """
    points_list = []
    for field in keypoints_fields:
        points = sample[field].keypoints  # Get keypoints from field
        for pt in points:
            keypoints = pt.points
            task_description = pt.task_description
            pt_id = pt.id
            pt_info = [keypoints, task_description, pt_id]
            points_list.append(pt_info)
    return points_list

def _serialize_transform_record(transform_record):
    """Serialize a transform record by converting PositionType objects to strings.
    
    Recursively traverses dictionaries and lists to convert all PositionType objects.

    Args:
        transform_record: Transform record containing PositionType objects

    Returns:
        dict: Serialized transform record with PositionType objects converted to strings
    """
    def replace_position_types(data):
        if isinstance(data, dict):
            return {k: replace_position_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [replace_position_types(item) for item in data]
        elif data.__class__.__name__ == "PositionType":
            return data.__repr__()
        return data
    
    return replace_position_types(transform_record)

def transform_sample(
    sample: "fo.core.sample.Sample",
    transforms: list[tuple[str, Callable[[np.ndarray], np.ndarray], dict]],
    *,
    label_fields: list[str] | bool = True,
    new_filepath: str | None = None,
    tags: list[str] | None = None,
    transform_record: dict | str | None = None,
) -> str:
    """
    Apply image transforms to `sample`, save the transformed image as a new file
    (in the same directory as the original by default), create a new fo.Sample
    copying requested labels/metadata, add it to the same dataset, and return
    the new sample id.

    Parameters
    ----------
    sample
        A FiftyOne Sample object (source sample to transform).
    transforms
        A list of transforms, each being a tuple:
            (transform_name: str,
             transform_fn: Callable[[image_array], image_array],
             params: dict)
        - `transform_fn(image, **params)` MUST return a NumPy image array (BGR).
        - Example: [("grayscale", apply_grayscale, {"keep_3_channels": True})]
    label_fields
        - If True: copy all label fields discovered on the sample (via your util).
        - If list[str]: copy only the specified label fields (e.g., ["detections","keypoints"]).
    new_filepath
        - If str: write the transformed image to this exact path.
        - If None: write into the same directory as `sample.filepath` using a generated
          filename derived from the original with a short unique suffix/hash.
    tags
        - Optional list of strings to add to the new sample (e.g., ["augmented"]).
        - If None: no extra tags are added.
    transform_record
        - Optional provenance info. Can be a dict (the util will serialize it) or
          an already-serialized string. The util must attach this to the new sample
          (e.g., `sample.metadata["transform"] = transform_record` or similar).

    Returns
    -------
    str
        The new sample's id (string) after it has been added to `sample._dataset`.

    Side effects
    ------------
    - Writes the transformed image file to disk.
    - Creates and adds a new fo.Sample to the same dataset as `sample`.
    - Copies over the requested label fields and their full attributes/metadata.
    - Copies sample-level metadata (e.g., sample.media, sample.metadata) to the new sample
      unless explicitly documented otherwise.
    - Attaches tags and transform_record to the new sample.

    Errors
    ------
    - Raise ValueError / IOError on file write/read failures.
    - Raise TypeError if `transforms` is malformed or a transform function does not return
      a valid NumPy image array.
    """

    # Load the original image
    image = cv2.imread(sample.filepath)
    
    # Apply all transforms sequentially
    for transform_name, transform_fn, params in transforms:
        image = transform_fn(image, **params)
    
    # Determine output filepath
    if new_filepath is None:
        # Generate new filename in same directory
        original_path = Path(sample.filepath)
        hash_suffix = _create_hash()
        new_filename = f"{original_path.stem}_{hash_suffix}{original_path.suffix}"
        new_filepath = str(original_path.parent / new_filename)
    
    # Write transformed image to disk
    cv2.imwrite(new_filepath, image)
    
    # Create new sample
    new_sample = fo.Sample(filepath=new_filepath)
    
    # Copy metadata if it exists
    if sample.metadata is not None:
        new_sample.metadata = sample.metadata.copy()
    
    # Determine which label fields to copy
    if label_fields is True:
        fields_to_copy = _get_label_fields(sample)
    else:
        fields_to_copy = label_fields or []
    
    # Copy label fields
    for field_name in fields_to_copy:
        if hasattr(sample, field_name) and sample[field_name] is not None:
            new_sample[field_name] = sample[field_name].copy()
    
    # Always copy application and platform fields if they exist
    for field_name in ["application", "platform"]:
        if hasattr(sample, field_name) and sample[field_name] is not None:
            new_sample[field_name] = sample[field_name]
    
    # Add tags if provided
    if tags:
        new_sample.tags = tags
    
    # Add transform record if provided
    if transform_record is not None:
        if not hasattr(new_sample, 'metadata') or new_sample.metadata is None:
            new_sample.metadata = fo.ImageMetadata()
        # Store transform record in metadata
        if isinstance(transform_record, dict):
            transform_record = _serialize_transform_record(transform_record)
        new_sample.metadata["transform"] = transform_record

    # Add new sample to dataset and return its ID
    sample._dataset.add_sample(new_sample)
    return new_sample.id
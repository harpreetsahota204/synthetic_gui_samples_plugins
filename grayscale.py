import fiftyone.operators as foo
import fiftyone.operators.types as types
from typing import Dict, Any
import cv2
import numpy as np

# Expected helpers you already implemented
from .utils import (
    _serialize_transform_record,
    transform_sample,
    _get_label_fields,
)


class GrayscaleAugment(foo.Operator):
    """
    Minimal GrayscaleAugment operator.

    - Operates on samples in ctx.selected if present, otherwise on ctx.view.
    - For each sample:
        * apply apply_grayscale(image) -> 3-channel BGR image
        * create a new fo.Sample saved in the same directory as the original
        * copy all original label fields (detections, keypoints, metadata, attributes)
        * attach a minimal transform_record for provenance (serialized)
    - Returns a minimal status dict on completion.
    """

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="grayscale_augment",
            label="Apply Grayscale Augmentation",
        )

    def resolve_input(self, ctx) -> types.Property:
        """
        Define user inputs that will be available in ctx.params during execute().
        """
        inputs = types.Object()
        
        # Let user choose which label fields to copy
        available_fields = _get_label_fields(ctx)
        if available_fields:
            for field_name in available_fields:
                inputs.bool(
                    f"copy_{field_name}",
                    default=True,
                    label=f"Copy {field_name}",
                    description=f"Copy {field_name} labels to the new grayscale samples",
                    view=types.CheckboxView()
                )
        else:
            # If no fields detected, provide default options
            inputs.bool(
                "copy_detections",
                default=True,
                label="Copy detections",
                description="Copy detection labels to the new grayscale samples",
                view=types.CheckboxView()
            )
            inputs.bool(
                "copy_keypoints", 
                default=True,
                label="Copy keypoints",
                description="Copy keypoint labels to the new grayscale samples",
                view=types.CheckboxView()
            )
        
        return types.Property(inputs, view=types.View(label="Grayscale Augmentation"))

    def execute(self, ctx) -> Dict[str, Any]:
        """
        - Determine samples: ctx.dataset.select(ctx.selected) if ctx.selected else ctx.view
        - Determine label fields to copy via _get_label_fields(ctx);
          if that helper returns nothing, fall back to ['detections','keypoints'].
        - For each sample, call transform_sample(...) with transforms = [("grayscale", apply_grayscale, {})]
          and new_filepath=None (util must save in same dir and create the new sample).
        - Do not yield stats — return a simple status.
        """
        
        def apply_grayscale(image: np.ndarray) -> np.ndarray:
            """Convert image to 3-channel BGR grayscale.
            
            Args:
                image: Input BGR image array
                
            Returns:
                np.ndarray: 3-channel BGR grayscale image
            """
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Convert back to 3-channel BGR (all channels same)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return gray_bgr
        
        # determine samples to process
        if getattr(ctx, "selected", None):
            samples = ctx.dataset.select(ctx.selected)
        else:
            samples = ctx.view

        # determine which label fields to copy based on user input
        available_fields = _get_label_fields(ctx)
        label_fields = []
        
        # Check user's checkbox selections from ctx.params
        for field_name in available_fields:
            if ctx.params.get(f"copy_{field_name}", False):
                label_fields.append(field_name)
        
        # If no available fields were detected, check default checkboxes
        if not available_fields:
            if ctx.params.get("copy_detections", False):
                label_fields.append("detections")
            if ctx.params.get("copy_keypoints", False):
                label_fields.append("keypoints")

        # minimal transform record for provenance
        transform_record = {"name": "grayscale", "params": {}, "plugin": "grayscale_augment"}
        serialized_transform = _serialize_transform_record(transform_record)

        # single transform: apply_grayscale must return a 3-channel BGR image
        transforms = [("grayscale", apply_grayscale, {})]

        # Apply transform to each sample and create a new sample with identical annotations
        for sample in samples:
            # transform_sample is expected to:
            #  - apply the callable transforms to the image,
            #  - save the new image in the same directory as original (new_filepath=None),
            #  - create an fo.Sample copying all requested label fields and metadata,
            #  - attach transform_record to the new sample,
            #  - add the new sample to the dataset and return its id (optional).
            transform_sample(
                sample,
                transforms,
                label_fields=label_fields,
                new_filepath=None,               # util saves next to original
                tags=None,                       # no tags in minimal form
                transform_record=serialized_transform,
            )

        # minimal response; no counters or errors returned
        return {"status": "success"}

    def resolve_output(self, ctx) -> types.Property:
        """
        Minimal output: a simple status.
        """
        outputs = types.Object()
        outputs.str("status", label="Status")
        return types.Property(outputs, view=types.View(label="Grayscale Augmentation Complete"))

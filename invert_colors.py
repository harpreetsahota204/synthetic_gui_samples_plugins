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
    _get_detections_fields,
    _get_keypoints_fields,
)

def _handle_calling(
        uri, 
        sample_collection, 
        copy_detections,
        copy_keypoints,        
        delegate
        ):
    ctx = dict(dataset=sample_collection)

    params = dict(
        copy_detections=copy_detections,
        copy_keypoints=copy_keypoints,
        delegate=delegate
    )
    return foo.execute_operator(uri, ctx, params=params)

def apply_color_inversion(image: np.ndarray) -> np.ndarray:
    """Invert image colors using cv2.bitwise_not."""
    return cv2.bitwise_not(image)

class InvertColorsAugment(foo.Operator):
    """

    - Operates on samples in ctx.selected if present, otherwise on ctx.view.
    - For each sample:
        * apply color inversion (cv2.bitwise_not) -> 3-channel BGR image
        * create a new fo.Sample saved in the same directory as the original
        * copy all original label fields (detections, keypoints, metadata, attributes)
        * attach a minimal transform_record for provenance (serialized)
    - Returns a minimal status dict on completion.
    """

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="invert_colors_augment",
            label="Apply Color Inversion Augmentation",
            icon="/assets/invert-svgrepo-com.svg",
        )

    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()
        
        # Check what label fields are actually present
        has_detections = "detections" in ctx.dataset.get_field_schema()
        has_keypoints = "keypoints" in ctx.dataset.get_field_schema()
        
        # Only show checkboxes for label types that actually exist
        if has_detections:
            inputs.bool(
                "copy_detections",
                default=True,
                label="Copy detections",
                description="Copy detection labels to the new inverted samples",
                view=types.CheckboxView()
            )
        
        if has_keypoints:
            inputs.bool(
                "copy_keypoints", 
                default=True,
                label="Copy keypoints",
                description="Copy keypoint labels to the new inverted samples",
                view=types.CheckboxView()
            )
        
        # Show a message if no supported label fields are found
        if not has_detections and not has_keypoints:
            inputs.view(
                "no_labels_notice",
                types.Notice(
                    label="No detections or keypoints found in the current samples. New samples will be created without labels."
                )
            )
        
        # Delegation option
        inputs.bool(
            "delegate",
            default=False,
            label="Delegate execution?",
            description="If you choose to delegate this operation you must first have a delegated service running",
            view=types.CheckboxView()
        )

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        """Implement this method if you want to programmatically *force*
        this operation to be delegated or executed immediately.

        Returns:
            whether the operation should be delegated (True), run
            immediately (False), or None to defer to
            `resolve_execution_options()` to specify the available options
        """
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        """Executes the actual operation based on the hydrated `ctx`.
        All operators must implement this method.

        Returns:
            an optional dict of results values
        """

        # Get parameters from user input
        copy_detections = ctx.params.get("copy_detections", False)
        copy_keypoints = ctx.params.get("copy_keypoints", False)
        
        # Build label fields list based on user selection
        label_fields = []
        if copy_detections:
            label_fields.append("detections")
        if copy_keypoints:
            label_fields.append("keypoints")
        
        # Determine samples to process
        if getattr(ctx, "selected", None):
            samples = ctx.dataset.select(ctx.selected)
        else:
            samples = ctx.view

        # Transform record for provenance
        transform_record = {"name": "color_inversion", "params": {}, "plugin": "invert_colors_augment"}
        serialized_transform = _serialize_transform_record(transform_record)

        # Apply color inversion transform to each sample
        transforms = [("color_inversion", apply_color_inversion, {})]
        
        for sample in samples:
            transform_sample(
                sample,
                transforms,
                label_fields=label_fields,
                new_filepath=None,
                tags=["color_inverted"],
                transform_record=serialized_transform,
            )

        # Reload dataset to show new samples
        ctx.ops.reload_dataset()
        
        return {"status": "success"}

    def resolve_output(self, ctx) -> types.Property:
        """
        Minimal output: a simple status.
        """
        outputs = types.Object()
        outputs.str("status", label="Status")
        return types.Property(outputs, view=types.View(label="Color Inversion Augmentation Complete"))
    
    def __call__(
        self, 
        sample_collection, 
        copy_detections,
        copy_keypoints,        
        delegate
        ):
        return _handle_calling(
            self.uri,
            sample_collection, 
            copy_detections,
            copy_keypoints,        
            delegate
            )
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

def apply_colorblind_filter(image: np.ndarray, colorblind_type: str) -> np.ndarray:
    """Apply colorblind simulation matrix to image."""
    # Define colorblind simulation matrices
    transforms = {
        # Complete color blindness (dichromacy)
        'deuteranopia': np.array([[0.625, 0.375, 0], [0.7, 0.3, 0], [0, 0.3, 0.7]]),
        'protanopia': np.array([[0.567, 0.433, 0], [0.558, 0.442, 0], [0, 0.242, 0.758]]),
        'tritanopia': np.array([[0.95, 0.05, 0], [0, 0.433, 0.567], [0, 0.475, 0.525]]),
        
        # Anomalous trichromacy (milder forms)
        'deuteranomaly': np.array([[0.8, 0.2, 0], [0.258, 0.742, 0], [0, 0.142, 0.858]]),
        'protanomaly': np.array([[0.817, 0.183, 0], [0.333, 0.667, 0], [0, 0.125, 0.875]]),
        'tritanomaly': np.array([[0.967, 0.033, 0], [0, 0.733, 0.267], [0, 0.183, 0.817]])
    }
    
    transform_matrix = transforms.get(colorblind_type, transforms['deuteranopia'])
    return cv2.transform(image, transform_matrix)

class ColorblindSimAugment(foo.Operator):
    """
    Minimal ColorblindSimAugment operator.

    - Operates on samples in ctx.selected if present, otherwise on ctx.view.
    - For each sample:
        * apply colorblind simulation matrix (deuteranopia/protanopia/tritanopia) -> 3-channel BGR image
        * create a new fo.Sample saved in the same directory as the original
        * copy all original label fields (detections, keypoints, metadata, attributes)
        * attach a minimal transform_record for provenance (serialized)
    - Returns a minimal status dict on completion.
    """

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="colorblind_sim_augment",
            label="Apply Colorblind Simulation Augmentation",
        )

    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()
        
        # Dropdown for colorblind type selection with descriptions
        colorblind_choices = types.DropdownView()
        
        # Complete color blindness (dichromacy)
        colorblind_choices.add_choice(
            "deuteranopia", 
            label="Deuteranopia (Green-blind)",
            description="Complete absence of green cones - difficulty distinguishing red and green"
        )
        colorblind_choices.add_choice(
            "protanopia", 
            label="Protanopia (Red-blind)", 
            description="Complete absence of red cones - difficulty perceiving red light"
        )
        colorblind_choices.add_choice(
            "tritanopia", 
            label="Tritanopia (Blue-blind)",
            description="Complete absence of blue cones - difficulty perceiving blue light"
        )
        
        # Anomalous trichromacy (milder forms)
        colorblind_choices.add_choice(
            "deuteranomaly",
            label="Deuteranomaly (Reduced Green)",
            description="Reduced green sensitivity - most common form (~5% of males)"
        )
        colorblind_choices.add_choice(
            "protanomaly",
            label="Protanomaly (Reduced Red)", 
            description="Reduced red sensitivity - milder red-green color blindness"
        )
        colorblind_choices.add_choice(
            "tritanomaly",
            label="Tritanomaly (Reduced Blue)",
            description="Reduced blue sensitivity - very rare blue-yellow color blindness"
        )
        
        inputs.enum(
            "colorblind_type",
            values=colorblind_choices.values(),
            default="deuteranopia",
            required=True,
            label="Colorblindness Type",
            description="Select the type of colorblindness to simulate",
            view=colorblind_choices,
        )
        
        # Check what label fields are actually present
        has_detections = "detections" in ctx.dataset.get_field_schema()
        has_keypoints = "keypoints" in ctx.dataset.get_field_schema()
        
        # Only show checkboxes for label types that actually exist
        if has_detections:
            inputs.bool(
                "copy_detections",
                default=True,
                label="Copy detections",
                description="Copy detection labels to the new colorblind simulation samples",
                view=types.CheckboxView()
            )
        
        if has_keypoints:
            inputs.bool(
                "copy_keypoints", 
                default=True,
                label="Copy keypoints",
                description="Copy keypoint labels to the new colorblind simulation samples",
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
        colorblind_type = ctx.params.get("colorblind_type", "deuteranopia")
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
        transform_record = {
            "name": "colorblind_simulation", 
            "params": {"type": colorblind_type}, 
            "plugin": "colorblind_sim_augment"
        }
        serialized_transform = _serialize_transform_record(transform_record)

        # Apply colorblind simulation transform to each sample
        transforms = [("colorblind_sim", apply_colorblind_filter, {"colorblind_type": colorblind_type})]
        
        for sample in samples:
            transform_sample(
                sample,
                transforms,
                label_fields=label_fields,
                new_filepath=None,
                tags=[colorblind_type],
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
        return types.Property(outputs, view=types.View(label="Colorblind Simulation Augmentation Complete"))

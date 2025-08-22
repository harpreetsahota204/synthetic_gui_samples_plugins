import fiftyone.operators as foo
import fiftyone.operators.types as types
from typing import Dict, Any
import cv2
import numpy as np

from .utils import (
    _serialize_transform_record,
    transform_sample,
)

# Common screen resolutions
SCREEN_RESOLUTIONS = {
    "1920x1080": (1920, 1080),   # Full HD
    "1366x768": (1366, 768),     # HD (common laptop/older displays)
    "1280x720": (1280, 720),     # HD
    "1440x900": (1440, 900),     # WXGA+ (legacy MacBook Air, some PCs)
    "1536x864": (1536, 864),     # HD+ (Surface Laptop, certain Windows laptops)
    "1600x900": (1600, 900),     # HD+ (mid-range monitors/laptops)
    "1280x1024": (1280, 1024),   # SXGA (older business monitors)
    "1680x1050": (1680, 1050),   # WSXGA+ (older widescreen monitors)
    "2560x1080": (2560, 1080),   # UltraWide FHD (ultrawide monitors)
    "2560x1440": (2560, 1440),   # QHD (high-end/gaming monitors)
    "3440x1440": (3440, 1440),   # UltraWide QHD (premium ultrawide)
    "3840x2160": (3840, 2160),   # 4K UHD (premium/professional displays)
    "5120x2880": (5120, 2880),   # 5K (niche professional monitors)
    "1024x768": (1024, 768),     # XGA (legacy screens/tablets)
    "1280x800": (1280, 800),     # WXGA (laptops/small monitors)
}

def apply_resize(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize image to target dimensions."""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

class ResizeOperator(foo.Operator):
    """
    Resize GUI screenshots to common screen resolutions.
    
    Since bounding boxes and keypoints use relative coordinates (0-1),
    they will automatically scale correctly with the resized images.
    """

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="resize_images",
            label="Resize Images to Screen Resolutions",
            description="Resize GUI screenshots to common screen resolutions",
            icon="/assets/resize-svgrepo-com.svg",
        )

    def resolve_input(self, ctx):
        """Define user inputs for the resize operation."""
        inputs = types.Object()
        
        # Resolution selection dropdown
        resolution_choices = types.DropdownView()
        for res_name, (width, height) in SCREEN_RESOLUTIONS.items():
            resolution_choices.add_choice(
                res_name,
                label=f"{res_name} ({width}×{height})",
                description=f"Resize to {width} × {height} pixels"
            )
        
        inputs.enum(
            "resolution",
            values=list(SCREEN_RESOLUTIONS.keys()),
            default="1920x1080",
            required=True,
            label="Target Resolution",
            description="Select the target screen resolution",
            view=resolution_choices,
        )
        
        # Custom resolution option
        inputs.bool(
            "use_custom",
            default=False,
            label="Use Custom Resolution",
            description="Enable to specify custom width and height",
            view=types.CheckboxView()
        )
        
        # Custom width/height (only show if custom is enabled)
        if ctx.params.get("use_custom", False):
            inputs.int(
                "custom_width",
                label="Custom Width",
                description="Custom width in pixels",
                required=True,
                default=1920,
                min=100,
                max=7680
            )
            inputs.int(
                "custom_height", 
                label="Custom Height",
                description="Custom height in pixels",
                required=True,
                default=1080,
                min=100,
                max=4320
            )
        
        # Label field options
        has_detections = "detections" in ctx.dataset.get_field_schema()
        has_keypoints = "keypoints" in ctx.dataset.get_field_schema()
        
        if has_detections:
            inputs.bool(
                "copy_detections",
                default=True,
                label="Copy detections",
                description="Copy detection labels to resized images",
                view=types.CheckboxView()
            )
        
        if has_keypoints:
            inputs.bool(
                "copy_keypoints", 
                default=True,
                label="Copy keypoints",
                description="Copy keypoint labels to resized images",
                view=types.CheckboxView()
            )
        
        # Delegation option
        inputs.bool(
            "delegate",
            default=False,
            label="Delegate execution?",
            description="If you choose to delegate this operation you must first have a delegated service running",
            view=types.CheckboxView()
        )

        return types.Property(inputs, view=types.View(label="Resize Images"))
    
    def resolve_delegation(self, ctx):
        """Handle delegation based on user choice."""
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        """Execute the resize operation."""
        # Get parameters
        use_custom = ctx.params.get("use_custom", False)
        
        if use_custom:
            target_width = ctx.params.get("custom_width", 1920)
            target_height = ctx.params.get("custom_height", 1080)
            resolution_name = f"{target_width}x{target_height}"
        else:
            resolution_name = ctx.params.get("resolution", "1920x1080")
            target_width, target_height = SCREEN_RESOLUTIONS[resolution_name]
        
        copy_detections = ctx.params.get("copy_detections", False)
        copy_keypoints = ctx.params.get("copy_keypoints", False)
        
        # Determine samples to process
        if getattr(ctx, "selected", None):
            samples = ctx.dataset.select(ctx.selected)
        else:
            samples = ctx.view

        # Transform record for provenance
        transform_record = {
            "name": "resize", 
            "params": {
                "target_width": target_width,
                "target_height": target_height,
                "resolution": resolution_name
            }, 
            "plugin": "resize_images"
        }
        serialized_transform = _serialize_transform_record(transform_record)

        # Process each sample
        for sample in samples:
            # Determine which label fields to copy
            label_fields = []
            if copy_detections:
                label_fields.append("detections")
            if copy_keypoints:
                label_fields.append("keypoints")
            
            # Create resize transform using module-level function
            transforms = [("resize", apply_resize, {"target_width": target_width, "target_height": target_height})]
            
            # Apply transform and create new sample
            new_sample_id = transform_sample(
                sample,
                transforms,
                label_fields=label_fields,
                new_filepath=None,
                tags=[f"resized_{resolution_name}"],
                transform_record=serialized_transform,
            )
            
            # Update metadata to reflect new image dimensions
            new_sample = ctx.dataset[new_sample_id]
            if new_sample.metadata is not None:
                new_sample.metadata.width = target_width
                new_sample.metadata.height = target_height
                new_sample.save()

        # Reload dataset to show new samples
        ctx.ops.reload_dataset()
        
        return {"status": "success"}

    def resolve_output(self, ctx) -> types.Property:
        """Simple output showing completion status."""
        outputs = types.Object()
        outputs.str("status", label="Status")
        return types.Property(outputs, view=types.View(label="Image Resize Complete"))

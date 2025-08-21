import fiftyone.operators as foo
import fiftyone.operators.types as types
from typing import Dict, Any
import cv2
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Expected helpers you already implemented
from .utils import (
    _serialize_transform_record,
    transform_sample,
    _get_label_fields,
    _get_detections_fields,
    _get_keypoints_fields,
)

# Prompt templates as constants
REPHRASE_PROMPT = """Generate a slight rephrasing of this UI task description. Keep the same meaning but use different words, style, tone, etc.

Original: "{text}"

Rephrased:"""

TRANSLATE_PROMPT = """Translate this UI task description to {target_language}. Keep the same meaning and technical accuracy.

Original: "{text}"

Translation:"""

def identity_transform(image: np.ndarray) -> np.ndarray:
    """Identity transform - returns image unchanged."""
    return image

def initialize_llm(model_name: str):
    """Initialize the LLM model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def rephrase_text(text: str, model, tokenizer, mode: str, target_language: str = "") -> str:
    """Use LLM to rephrase or translate text."""
    if mode == "translate" and target_language:
        prompt = TRANSLATE_PROMPT.format(text=text, target_language=target_language)
    else:
        prompt = REPHRASE_PROMPT.format(text=text)
    
    # Standard generation for all models
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    
    # Use model card recommended parameters
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        temperature=0.6,
        do_sample=True,
        top_p=0.95,
        top_k=20,
    )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    # Clean up the response
    lines = content.strip().split('\n')
    rephrased = lines[0].strip().strip('"').strip("'")
    
    # Fallback to original if something went wrong
    if not rephrased or len(rephrased) < 3:
        return text
        
    return rephrased

class TaskDescriptionAugment(foo.Operator):
    """
    LLM-based TaskDescriptionAugment operator.

    - Operates on samples in ctx.selected if present, otherwise on ctx.view.
    - For each sample:
        * copy the image as-is (no visual transformation)
        * iterate through detections/keypoints and rephrase task_description attributes using LLM
        * preserve original text under original_task_description
        * create a new fo.Sample with modified task descriptions
        * attach LLM transform provenance
    - Returns a minimal status dict on completion.
    """

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="task_description_augment",
            label="Rephrase Task Descriptions with LLM",
            dynamic=True,  # Enable dynamic inputs for conditional language field
        )

    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()
        
        # Model selection dropdown
        model_choices = types.DropdownView()
        model_choices.add_choice(
            "Qwen/Qwen3-0.6B",
            label="Qwen3-0.6B (Fastest, smallest)",
            description="Lightweight model for quick rephrasing"
        )
        model_choices.add_choice(
            "Qwen/Qwen3-1.7B", 
            label="Qwen3-1.7B (Balanced)",
            description="Good balance of speed and quality"
        )
        model_choices.add_choice(
            "Qwen/Qwen3-8B-MLX-bf16",
            label="Qwen3-8B-MLX (High Quality)",
            description="Requires: pip install mlx_lm"
        )
        model_choices.add_choice(
            "Qwen/Qwen3-1.7B-MLX-bf16",
            label="Qwen3-1.7B-MLX (Balanced MLX)",
            description="Requires: pip install mlx_lm"
        )
        
        inputs.enum(
            "model_name",
            values=model_choices.values(),
            default="Qwen/Qwen3-1.7B",
            required=True,
            label="LLM Model",
            description="Select the language model for rephrasing task descriptions",
            view=model_choices,
        )
        
        # Mode selection: simple rephrasing vs translation
        mode_choices = types.RadioGroup()
        
        mode_choices.add_choice(
            "rephrase",
            label="Simple Rephrasing",
            description="Rephrase task descriptions while keeping the same language"
        )
        mode_choices.add_choice(
            "translate",
            label="Translate to Different Language",
            description="Translate task descriptions to a different language"
        )
        
        inputs.enum(
            "mode",
            values=mode_choices.values(),
            default="rephrase",
            required=True,
            label="Processing Mode",
            view=mode_choices,
        )
        
        # Dynamic language input - only show for translation mode
        mode = ctx.params.get("mode")
        if mode == "translate":
            inputs.str(
                "target_language",
                label="Target Language",
                description="What language do you want to translate to?",
                required=True,
                default="Punjabi"
            )
        
        # Check what label fields are actually present
        has_detections = "detections" in ctx.dataset.get_field_schema()
        has_keypoints = "keypoints" in ctx.dataset.get_field_schema()
        
        # Only show checkboxes for label types that actually exist
        if has_detections:
            inputs.bool(
                "process_detections",
                default=True,
                label="Process detection task descriptions",
                description="Rephrase/translate task_description attributes in detection labels",
                view=types.CheckboxView()
            )
        
        if has_keypoints:
            inputs.bool(
                "process_keypoints", 
                default=True,
                label="Process keypoint task descriptions",
                description="Rephrase/translate task_description attributes in keypoint labels",
                view=types.CheckboxView()
            )
        
        # Show a message if no supported label fields are found
        if not has_detections and not has_keypoints:
            inputs.view(
                "no_labels_notice",
                types.Notice(
                    label="No detections or keypoints found. This operator requires labels with task_description attributes."
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
        model_name = ctx.params.get("model_name", "Qwen/Qwen3-1.7B")
        mode = ctx.params.get("mode", "rephrase")
        target_language = ctx.params.get("target_language", "")
        process_detections = ctx.params.get("process_detections", False)
        process_keypoints = ctx.params.get("process_keypoints", False)
        
        # Initialize LLM with error handling
        try:
            ctx.ops.notify(f"Initializing {model_name}...")
            model, tokenizer = initialize_llm(model_name)
            ctx.ops.notify(f"Model {model_name} loaded successfully")
        except Exception as e:
            error_msg = f"Failed to initialize LLM model '{model_name}': {str(e)}"
            ctx.ops.notify(error_msg, kind="error")
            return {"status": "error", "message": error_msg}
        
        # Determine samples to process
        if getattr(ctx, "selected", None):
            samples = ctx.dataset.select(ctx.selected)
        else:
            samples = ctx.view

        # Transform record for provenance
        transform_record = {
            "name": "task_description_augment", 
            "params": {
                "model": model_name,
                "mode": mode,
                "target_language": target_language if mode == "translate" else None
            }, 
            "plugin": "task_description_augment"
        }
        serialized_transform = _serialize_transform_record(transform_record)

        # Initialize LLM
        model, tokenizer = initialize_llm(model_name)
        
        # Process each sample
        for sample in samples:
            # Determine which label fields to copy
            label_fields = []
            if process_detections:
                label_fields.append("detections")
            if process_keypoints:
                label_fields.append("keypoints")
            
            # Create the new sample using transform_sample (identity transform - image unchanged)
            transforms = [("identity", identity_transform, {})]
            new_sample_id = transform_sample(
                sample,
                transforms,
                label_fields=label_fields,
                new_filepath=None,
                tags=["task_description_augmented"],
                transform_record=serialized_transform,
            )
            
            # Get the newly created sample and modify its task descriptions
            new_sample = ctx.dataset[new_sample_id]
            
            # Process detections if requested
            if process_detections and hasattr(new_sample, "detections") and new_sample.detections is not None:
                for detection in new_sample.detections.detections:
                    if hasattr(detection, "task_description") and detection.task_description:
                        # Preserve original task description
                        detection.original_task_description = detection.task_description
                        # Generate new task description using LLM
                        detection.task_description = rephrase_text(
                            detection.task_description, 
                            model, 
                            tokenizer, 
                            mode, 
                            target_language
                        )
            
            # Process keypoints if requested
            if process_keypoints and hasattr(new_sample, "keypoints") and new_sample.keypoints is not None:
                for keypoint in new_sample.keypoints.keypoints:
                    if hasattr(keypoint, "task_description") and keypoint.task_description:
                        # Preserve original task description
                        keypoint.original_task_description = keypoint.task_description
                        # Generate new task description using LLM
                        keypoint.task_description = rephrase_text(
                            keypoint.task_description, 
                            model, 
                            tokenizer, 
                            mode, 
                            target_language
                        )
            
            # Save the modified sample
            new_sample.save()

        # Reload dataset to show new samples
        ctx.ops.reload_dataset()
        
        return {"status": "success"}

    def resolve_output(self, ctx) -> types.Property:
        """
        Simple output: just show status.
        """
        outputs = types.Object()
        outputs.str("status", label="Status")
        return types.Property(outputs, view=types.View(label="Task Description Augmentation Complete"))
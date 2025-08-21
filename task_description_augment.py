import fiftyone.operators as foo
import fiftyone.operators.types as types
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TaskDescriptionAugment(foo.Operator):
    """Use an LLM to rephrase detection 'task_description' attributes."""

    def __init__(self, model_name="Qwen/Qwen3-1.7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    @property
    def config(self):
        """Defines metadata about the operator."""
        return foo.OperatorConfig(
            name="task_description_augment",
            label="Rephrase Task Descriptions",
        )

    def resolve_input(self, ctx):
        """Defines the UI for input parameters."""
        inputs = types.Object()
        inputs.str("detections_field", label="Detections field", default="detections")
        inputs.str("attribute", label="Attribute to rephrase", default="task_description")
        inputs.str("action_type_field", label="Action type field", default="action_type")

        header = "Task Description Augmentation"
        return types.Property(inputs, view=types.View(label=header))

    def execute(self, ctx):
        """Core logic of the operator."""
        sample_collection = ctx.sample_collection
        detections_field = ctx.inputs.detections_field
        attribute = ctx.inputs.attribute
        action_type_field = ctx.inputs.action_type_field

        for sample in sample_collection:
            detections = sample[detections_field]
            for detection in detections:
                if attribute in detection.attributes:
                    original_task = detection.attributes[attribute]
                    action_type = detection.attributes.get(action_type_field)

                    # Generate rephrased task description using LLM
                    prompt = f"""Rephrase the following task description: "{original_task}"
                    Action type: {action_type if action_type else 'N/A'}"""
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9
                    )
                    rephrased_task = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

                    # Update detection attributes
                    detection.attributes[attribute] = rephrased_task
                    detection.attributes["original_task_description"] = original_task

            sample.save()

        return {"processed": len(sample_collection), "detections_field": detections_field}

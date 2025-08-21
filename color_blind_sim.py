import fiftyone.operators as foo
import fiftyone.operators.types as types
import cv2
import numpy as np

class ColorblindSimAugment(foo.Operator):
    """
    Apply a colorblind simulation filter (deuteranopia, protanopia, tritanopia) to images in the dataset/view.
    """

    @property
    def config(self):
        """
        Defines metadata about the operator: its name and label as shown in the FiftyOne UI.
        """
        return foo.OperatorConfig(
            name="colorblind_sim_augment",
            label="Apply Colorblind Simulation Augmentation",
        )

    def resolve_input(self, ctx):
        """
        Defines the UI for input parameters.

        Expected:
        - A dropdown for selecting the colorblindness type (deuteranopia, protanopia, tritanopia)
        - A text field for the new field name (e.g., 'colorblind_sim')
        - Possibly a toggle for overwriting if the field already exists
        """
        inputs = types.Object()
        inputs.str("colorblind_type", label="Colorblindness Type", choices=["deuteranopia", "protanopia", "tritanopia"], default="deuteranopia")
        inputs.str("field_name", label="Output field name", required=True)
        inputs.bool("overwrite", label="Overwrite existing", default=False)

        header = "Colorblind Simulation Augmentation"
        return types.Property(inputs, view=types.View(label=header))

    def execute(self, ctx):
        """
        Core logic of the operator.

        Steps:
        - Get dataset/view from context
        - For each selected sample:
            - Load image
            - Apply colorblind simulation (cv2 or custom matrix)
            - Save augmented version to disk
            - Add path reference to sample[field_name]
        - Save dataset

        Returns:
            dict: summary info (e.g., num_samples_processed)
        """
        # Define colorblind simulation matrices
        transforms = {
            'deuteranopia': np.array([[0.625, 0.375, 0], [0.7, 0.3, 0], [0, 0.3, 0.7]]),
            'protanopia': np.array([[0.567, 0.433, 0], [0.558, 0.442, 0], [0, 0.242, 0.758]]),
            'tritanopia': np.array([[0.95, 0.05, 0], [0, 0.433, 0.567], [0, 0.475, 0.525]])
        }

        # Retrieve operator parameters
        colorblind_type = ctx.inputs.colorblind_type
        field_name = ctx.inputs.field_name
        overwrite = ctx.inputs.overwrite

        # Validate colorblind type
        if colorblind_type not in transforms:
            raise ValueError(f"Unsupported colorblind type: {colorblind_type}")

        # Get dataset/view from context
        sample_collection = ctx.dataset

        # Initialize counters
        num_samples_processed = 0

        # Process each sample
        for sample in sample_collection:
            # Load image
            image = cv2.imread(sample.filepath)
            if image is None:
                continue

            # Apply colorblind simulation
            sim_image = cv2.transform(image, transforms[colorblind_type])

            # Save augmented image
            output_path = f"{sample.filepath.rsplit('.', 1)[0]}_{colorblind_type}.jpg"
            cv2.imwrite(output_path, sim_image)

            # Update sample with new field
            sample[field_name] = output_path
            sample.save()

            num_samples_processed += 1

        return {"processed": num_samples_processed, "field_name": field_name}

    def resolve_output(self, ctx):
        """
        Defines what to show in the UI after execution.

        Example outputs:
        - Number of samples processed
        - Name of field where augmentation was stored
        """
        outputs = types.Object()
        outputs.int("processed", label="Samples processed")
        outputs.str("field_name", label="Output field name")

        header = "Colorblind Simulation Augmentation Complete"
        return types.Property(outputs, view=types.View(label=header))

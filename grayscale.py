import fiftyone.operators as foo
import fiftyone.operators.types as types

class GrayscaleAugment(foo.Operator):
    """
    Operator that applies a grayscale transformation to selected samples
    in a FiftyOne dataset.

    Inputs:
        - View/Dataset selection (sample IDs or current view)
        - Output field name (where augmented image will be stored)

    Execution:
        - Reads each selected sample's image
        - Converts image to grayscale (keeping 3-channel consistency)
        - Writes augmented image back to dataset under new field

    Outputs:
        - Confirmation message
        - Number of samples successfully augmented
    """

    @property
    def config(self):
        """
        Defines metadata about the operator: its name and label as shown
        in the FiftyOne UI.
        """
        return foo.OperatorConfig(
            name="grayscale_augment",
            label="Apply Grayscale Augmentation",
        )

    def resolve_input(self, ctx):
        """
        Defines the UI for input parameters.

        Expected:
        - A text field for the new field name (e.g. 'grayscale')
        - Possibly a toggle for overwriting if the field already exists
        """
        inputs = types.Object()
        inputs.str("field_name", label="Output field name", required=True)
        inputs.bool("overwrite", label="Overwrite existing", default=False)

        header = "Grayscale Augmentation"
        return types.Property(inputs, view=types.View(label=header))

    def execute(self, ctx):
        """
        Core logic of the operator.

        Steps:
        - Get dataset/view from context
        - For each selected sample:
            - Load image
            - Convert to grayscale (cv2 or PIL)
            - Save augmented version to disk
            - Add path reference to sample[field_name]
        - Save dataset

        Returns:
            dict: summary info (e.g., num_samples_processed)
        """
        pass

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

        header = "Grayscale Augmentation Complete"
        return types.Property(outputs, view=types.View(label=header))

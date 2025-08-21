import fiftyone.operators as foo
import fiftyone.operators.types as types
import cv2

class InvertColorsAugment(foo.Operator):
    """
    Operator that inverts the colors of images in the dataset/view.

    Inputs:
        - View/Dataset selection (sample IDs or current view)
        - Output field name (where augmented image will be stored)

    Execution:
        - Reads each selected sample's image
        - Inverts the image colors
        - Writes the augmented image back to the dataset under the specified field

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
            name="invert_colors_augment",
            label="Apply Color Inversion Augmentation",
        )

    def resolve_input(self, ctx):
        """
        Defines the UI for input parameters.

        Expected:
        - A text field for the new field name (e.g., 'inverted_colors')
        - Possibly a toggle for overwriting if the field already exists
        """
        inputs = types.Object()
        inputs.str("field_name", label="Output field name", required=True)
        inputs.bool("overwrite", label="Overwrite existing", default=False)

        header = "Color Inversion Augmentation"
        return types.Property(inputs, view=types.View(label=header))

    def execute(self, ctx):
        """
        Core logic of the operator.

        Steps:
        - Get dataset/view from context
        - For each selected sample:
            - Load image
            - Invert colors (cv2.bitwise_not or other method)
            - Save augmented version to disk
            - Add path reference to sample[field_name]
        - Save dataset

        Returns:
            dict: summary info (e.g., num_samples_processed)
        """
        # TODO: Implement the core logic for color inversion augmentation
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

        header = "Color Inversion Augmentation Complete"
        return types.Property(outputs, view=types.View(label=header))

# Synthetic GUI Samples Plugin for FiftyOne

A comprehensive FiftyOne plugin for augmenting GUI screenshot datasets with computer vision and language model transformations. This plugin provides a collection of operators designed specifically for synthetic GUI data generation and augmentation.

## 🎯 Overview

This plugin is designed for image datasets containing GUI screenshots in [COCO4GUI format](https://github.com/harpreetsahota204/coco4gui_fiftyone), with `detections` (bounding boxes) and `keypoints` annotations. It provides both visual transformations and text augmentation capabilities to create diverse synthetic training data for GUI understanding models.

### Key Features

- **Visual Augmentations**: Grayscale conversion, color inversion, colorblind simulation, and resolution scaling

- **LLM-Powered Text Augmentation**: Rephrase or translate task descriptions using local language models

- **Annotation Preservation**: All transformations preserve bounding boxes, keypoints, and metadata

- **Flexible Resolution Support**: Resize images to common screen resolutions for multi-device training

- **Provenance Tracking**: Complete transformation history for reproducibility

## 📦 Installation

### Prerequisites

- FiftyOne >= 1.3.0
- Python >= 3.10
- OpenCV (`cv2`)
- NumPy
- Pillow (PIL)

### For LLM Text Augmentation

```bash
# Standard PyTorch installation
pip install torch transformers
```

### Plugin Installation

1. Download this repository to your FiftyOne plugins directory:

```bash
fiftyone plugins download https://github.com/harpreetsahota204/synthetic_gui_samples_plugins.git
```

2. Restart your FiftyOne App to load the plugin:

```bash
# If FiftyOne is running, restart it
fiftyone app launch
```

## 🚀 Operators

This plugin provides five main operators accessible through the FiftyOne App's operator browser:

### 1. Grayscale Augmentation

**Operator**: `grayscale_augment`

Converts images to 3-channel grayscale while preserving all annotations.

**Features**:

- Maintains 3-channel BGR format for compatibility

- Preserves all bounding boxes and keypoints

- Optional label field copying controls

**Use Case**: Create grayscale variants for robustness testing and data diversity.

### 2. Color Inversion Augmentation

**Operator**: `invert_colors_augment`

Inverts image colors using bitwise NOT operation.

**Features**:

- Complete color inversion (white becomes black, etc.)

- Preserves spatial relationships

- Maintains annotation accuracy

**Use Case**: Test model robustness to inverted color schemes (dark mode UIs, high contrast displays).

### 3. Colorblind Simulation

**Operator**: `colorblind_sim_augment`

Simulates various types of color vision deficiency.

**Supported Types**:

- **Deuteranopia**: Green-blind (complete)

- **Protanopia**: Red-blind (complete)  

- **Tritanopia**: Blue-blind (complete)

- **Deuteranomaly**: Green-weak (partial)

- **Protanomaly**: Red-weak (partial)

- **Tritanomaly**: Blue-weak (partial)

**Use Case**: Ensure GUI accessibility by testing how interfaces appear to users with color vision deficiencies.

### 4. Task Description Augmentation

**Operator**: `task_description_augment`

Uses local language models to rephrase or translate task descriptions in annotations.

**Features**:

- **Multiple Models**: Support for Qwen3-0.6B, Qwen3-1.7B

- **Two Modes**:

  - **Rephrase**: Generate alternative phrasings in the same language

  - **Translate**: Convert to different languages

- **Provenance**: Preserves original text and includes reasoning

- **Selective Processing**: Choose which annotation types to process

**Use Case**: Create diverse language variations for multilingual GUI understanding or paraphrase augmentation.

### 5. Resolution Scaling

**Operator**: `resize_images`

Resizes images to common screen resolutions while maintaining annotation accuracy.

**Supported Resolutions**:

- **Mobile/Tablet**: 1024×768, 1280×800

- **Laptop/Desktop**: 1366×768, 1920×1080, 1440×900, 1536×864

- **High-End**: 2560×1440, 3840×2160 (4K), 5120×2880 (5K)

- **Ultrawide**: 2560×1080, 3440×1440

- **Custom**: User-defined dimensions

**Features**:

- Automatic annotation scaling (relative coordinates preserved)

- Multiple interpolation methods

- Batch processing support

**Use Case**: Generate training data for different screen sizes and device types.

## 📋 Usage

### Basic Workflow

1. **Load your dataset** in FiftyOne App

2. **Select samples** (optional) - operators work on selection or entire view

3. **Open operator browser** (⚡ icon in the App)

4. **Choose an operator** from the `@harpreetsahota/synthetic_gui_samples_plugins` section

5. **Configure parameters** in the operator form

6. **Execute** immediately or delegate for background processing

### Example: Grayscale Augmentation

```python
import fiftyone as fo

# Load your GUI dataset
dataset = fo.load_dataset("my_gui_dataset")

# Launch FiftyOne App
session = fo.launch_app(dataset)

# In the App:
# 1. Select samples or use entire view
# 2. Open operator browser (⚡ icon)
# 3. Find "Apply Grayscale Augmentation"
# 4. Choose which label fields to copy
# 5. Execute
```

### Example: LLM Text Augmentation

```python
# For task description rephrasing:
# 1. Select samples with task_description annotations
# 2. Run "Rephrase Task Descriptions with LLM"
# 3. Choose model (Qwen3-0.6B for speed, Qwen3-1.7B for quality)
# 4. Select "Simple Rephrasing" mode
# 5. Choose annotation types to process
# 6. Execute

# For translation:
# 1. Same as above but select "Translate to Different Language"
# 2. Specify target language (e.g., "Spanish", "French", "Chinese")
# 3. Execute
```

### Example: Resolution Scaling

```python
# Resize to common resolutions:
# 1. Select GUI screenshots
# 2. Run "Resize Images to Screen Resolutions"  
# 3. Choose target resolution (e.g., "1920x1080")
# 4. Or enable custom resolution and specify dimensions
# 5. Select which annotations to preserve
# 6. Execute
```

## 🔧 Configuration

### Execution Modes

All operators support two execution modes:

- **Immediate**: Process immediately in the FiftyOne App (default)

- **Delegated**: Queue for background processing (requires orchestrator setup)

### Label Field Selection

Most operators allow you to choose which annotation fields to copy:

- **Detections**: Bounding box annotations

- **Keypoints**: Point-based annotations  

- **All Fields**: Copy all label fields automatically

### Output Location

Transformed images are saved in the same directory as original images with unique hash suffixes to prevent conflicts.

## 🏗️ Architecture

### Core Components

- **`transform_sample()`**: Central utility for applying image transformations

- **Transform Functions**: OpenCV-based image processing functions

- **LLM Integration**: Hugging Face Transformers for text processing

- **Annotation Handling**: Automatic copying and scaling of spatial annotations

### Transform Record

Each augmented sample includes a `transform_record` in its metadata for full provenance tracking:

```python
{
    "transforms": [{"name": "grayscale", "params": {}}],
    "source_sample_id": "original_id",
    "timestamp": "2024-01-01T12:00:00",
    "plugin": "synthetic_gui_samples_plugins"
}
```

## 🎨 Supported Annotation Types

- **Detections** (`fo.Detections`): Bounding boxes with labels and attributes

- **Keypoints** (`fo.Keypoints`): Point-based annotations

- **Task Descriptions**: Text attributes on detection/keypoint objects

- **Custom Attributes**: All custom fields and metadata are preserved

## 🔍 Use Cases

### GUI Model Training

- **Multi-Resolution Training**: Generate samples at different screen resolutions

- **Accessibility Testing**: Create colorblind-simulated variants

- **Robustness Testing**: Test with inverted colors and grayscale images

- **Multilingual Support**: Generate translated task descriptions

### Data Augmentation Pipeline

```python
# Example workflow combining multiple operators:
# 1. Start with base GUI screenshots
# 2. Apply grayscale augmentation for robustness
# 3. Resize to multiple resolutions for device compatibility  
# 4. Use colorblind simulation for accessibility
# 5. Rephrase task descriptions for linguistic diversity
```

### Research Applications

- **Vision-Language Models**: Train on diverse visual and textual variations

- **Accessibility Research**: Study GUI perception across different visual conditions

- **Cross-Cultural UX**: Generate multilingual interface descriptions

## ⚙️ Advanced Features

### LLM Models

The plugin supports multiple language models with different performance characteristics:

| Model | Size | Speed | Quality | 
|-------|------|-------|---------|
| Qwen3-0.6B | Small | Fastest | Good |
| Qwen3-1.7B | Medium | Fast | Better |

### Custom Resolutions

Beyond predefined screen resolutions, you can specify custom dimensions for specialized use cases.

### Batch Processing

All operators support batch processing of selected samples or entire dataset views.

## 🛠️ Development

### Plugin Structure

```
synthetic_gui_samples_plugins/
├── fiftyone.yml              # Plugin configuration
├── __init__.py              # Plugin registration
├── utils.py                 # Core transformation utilities
├── grayscale.py            # Grayscale augmentation operator
├── invert_colors.py        # Color inversion operator
├── color_blind_sim.py      # Colorblind simulation operator
├── task_description_augment.py  # LLM text augmentation
├── resizer.py              # Resolution scaling operator
└── README.md               # This file
```

## 📝 Requirements

### Python Dependencies

```txt
fiftyone>=1.3.0
opencv-python
numpy
Pillow
torch
transformers
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.


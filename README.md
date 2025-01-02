# Neural Style Transfer  

## Introduction
This project implements a Neural Style Transfer model to apply artistic styles to photographs. It leverages a pre-trained VGG19 model to extract content and style features, combining them to generate stylized images.

## Instructions
1. Ensure Python 3.x and required libraries are installed.
2. Install dependencies using the command:
   ```bash
   pip install torch torchvision matplotlib Pillow
   ```
3. Place your content and style images in the project directory, naming them 'content.jpg' and 'style.jpg', respectively.
4. Run the script using the command:
   ```bash
   python neural_style_transfer.py
   ```
5. The output will be displayed and saved as a styled image.

## Requirements
- Python 3.x
- Libraries:
  - torch
  - torchvision
  - matplotlib
  - Pillow
- Pre-trained VGG19 model (downloaded automatically by torchvision)

## Output Example
The program generates a stylized image combining the content and style of the input images, producing artistic results.

## File Structure
```
|-- neural_style_transfer.py
|-- content.jpg
|-- style.jpg
|-- output.jpg
|-- README.md
```

## Notes
- The script uses an iterative optimization process to refine the stylized image. Adjust the number of steps and weights for style and content to fine-tune the output.
- Ensure input images are resized appropriately to avoid memory issues.




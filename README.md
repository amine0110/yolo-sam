# Yolo-SAM for Medical Imaging

This is a project that creates an AI-assisted medical image annotation tool using PyTorch, tkinter, pydicom, and other libraries. The tool allows users to import DICOM images, create and manipulate bounding boxes, and save their annotations. Additionally, it integrates with an AI model to automatically generate segmentation masks.

The official repo of the SAM model can be found [here](https://github.com/facebookresearch/segment-anything)

https://github.com/amine0110/yolo-sam/assets/37108394/c78441ae-ce76-4aa2-b3e1-06bce4b377d7

## Features
- Import and visualize DICOM images.
- Create, edit, and remove bounding boxes.
- Automatically generate segmentation masks using an AI model.
- Save segmentation masks as DICOM files.

## Prerequisites
- Python 3.x
- PyTorch
- tkinter
- ttkbootstrap
- pydicom
- dicom2nifti
- pillow
- torchvision

## Usage
- Use the `Open Directory` button to select a folder containing DICOM images.
- Navigate through the images using the "Previous" and "Next" buttons.
- Click the `Activate Draw` button and draw a bounding box on the image.
- To remove a bounding box, select it and click the `Delete Annotation` button.
- Click the `Launch SAM` button to generate segmentation masks. The AI model will predict the areas of interest within the bounding boxes.
- You can save the segmentation masks as DICOM files by clicking the `Export Segmentation` button. For each image, the tool will create a new DICOM file with the "_seg" suffix. If an image doesn't have any bounding boxes or masks, the tool will create an empty DICOM file.

## Contribution Guidelines
We greatly appreciate any contributions to the AI-Assisted Medical Image Annotation Tool project. Your input can make this tool more useful and effective!

Here are some guidelines to help you get started:

### Reporting Bugs
If you encounter any bugs or issues, please create an issue on GitHub with a detailed description of the problem. Include as much information as possible, such as the steps to reproduce the issue, the expected behavior, and the actual behavior. If applicable, add screenshots to illustrate the problem.

### Suggesting Enhancements
If you have an idea for a new feature or an improvement to an existing feature, we'd love to hear about it! Create an issue on GitHub to describe your idea. Please provide as much detail as possible, such as the expected benefits and any potential challenges.

### Making Changes
If you want to write code to fix a bug or add a new feature, here are the general steps:

- Fork the repository on GitHub.
- Create a new branch for your changes.
- Write your code.
- Make sure your code follows our style guide and is well-commented.
- Test your code to ensure it works as expected and doesn't introduce new bugs.
- Create a pull request detailing your changes. We'll review your pull request as soon as possible and provide feedback.
- Please note that we may not accept all pull requests, but we appreciate every contribution!

---
## 📩 Newsletter
Stay up-to-date on the latest in computer vision and medical imaging! Subscribe to my newsletter now for insights and analysis on the cutting-edge developments in this exciting field.

https://pycad.co/join-us/

---

## 📖 Medical Imaging E-Book
This ebook serves as a guide for those who are new to the profession of medical imaging. It provides definitions and resources like where to learn anatomy. Where can you find quality papers? Where can you find good, cost-free datasets? plus more.

Grab it from [here](https://pycad.co/medical-imaging-ebook/).

---
## 🏫 Courses

| Title | Tags | Link |
| --- | --- | --- |
| Python for Medical Imaging | `Dicom` `NIFTI` `ITK` `SimpleITK` `3D` `Python` | [Udemy](https://www.udemy.com/course/python-programming-for-medical-imaging/?referralCode=4EB87F3DE56679A11DA8) |
| How to Work With Dicom Using Python | `Dicom` `Medical Imaging` `Python` | [Udemy](https://www.udemy.com/course/how-to-work-with-dicom-using-python/?referralCode=ECBFF2BA3DED3608BE91) |
| How to Improve Medical Image Classification Results | `Medical Imaging` `Image Classification` `Python` | [YouTube](https://youtu.be/IXJMNGiBWy4) | 
| Automatic Liver Segmentation Using PyTorch and Monai | `Medical Imaging` `Image Segmentation` `Python` | [YouTube](https://youtu.be/AU4KlXKKnac) |
| Learn Tkinter from Scratch to Create Desktop Applications | `Python` `Tkinter` `GUI` | [YouTube](https://youtu.be/Fv82RX4cWW4) |
| Learn C++ for Beginners | `C++` `Basics` | [YouTube](https://youtu.be/94T4RQiD4Lo) |

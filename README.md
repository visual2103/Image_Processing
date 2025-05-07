# Color Image Segmentation Using Region Growing

![image](https://github.com/user-attachments/assets/88391d4e-2b2b-45a1-93fe-df24916d8107)
## Project Overview

This project aims to segment a color image into multiple regions based on color similarity. The goal is to group pixels with similar colors into coherent areas using the **Region Growing** algorithm. This method is useful for image analysis, object detection, and handling noisy or poorly delimited images.

---

## Motivation

Color images are everywhere — from smartphone photos to medical imaging and video surveillance. For machines to understand an image, it must first be segmented into meaningful parts. Region Growing is a simple yet effective approach that grows regions starting from seed pixels and adds similar neighbors based on a defined threshold.

---

## Methodology

### Preprocessing
1. **Gaussian Blur** – smooths the image and reduces noise using a 5x5 kernel.
2. **Color Space Transformation** – converts image from BGR to L\*u\*v\* using OpenCV’s `cvtColor()`.
3. **Standard Deviation** – computes standard deviations of the u and v channels to determine threshold `T`.

### Region Growing Algorithm
- Initialize a labels matrix (`CV_32SC1`) to keep track of region IDs.
- For each unvisited pixel, treat it as a seed and compute its local neighborhood mean in u and v channels.
- Add neighboring pixels to the current region if their Euclidean distance in (u, v) space is below threshold `T = scale × sqrt(std_u² + std_v²)`.
- Use a FIFO queue to process neighbors iteratively.
- Assign each region a unique label.

### Postprocessing
- **Erosion (×3)**: Removes noise and thin borders by shrinking regions.
- **Dilation**: Fills small gaps and expands regions to improve visual consistency.
- Repeat until all pixels are labeled.

---

## Visualization
- **`visualizeLabels()`** assigns random colors to regions for easy visual inspection.
- **`makeSegmented()`** computes the average color per region and builds a smooth segmented output image.

---

## Usage Guide

1. Run the application and select option **13 - Region Growing**.
2. Load a color BMP image.
3. Input a scale factor > 0 (e.g., `1.0`, `0.5`, `3.0`) to control segmentation granularity.
4. The application displays:
   - Original image
   - Blurred image
   - Randomly colored label image
   - Final segmented image after postprocessing

---

## Results

### Dataset
Tested on standard color BMP images (e.g., Lena, shapes) available [here](https://drive.google.com/drive/folders/1sFdJFAijbdoQht4N6fao_huZyBdf88mP).

### Observations
- `scale = 0.5` → more fine-grained regions
- `scale = 3.0` → larger, smoother regions
- Morphological postprocessing cleans noisy regions and fills holes

---

## Conclusions

- Successfully implemented a full Region Growing pipeline
- Achieved dynamic segmentation by tuning a single `scale` parameter
- Chose the L\*u\*v\* color space for perceptual accuracy
- Used OpenCV for color conversions and performance

---

## Future Work

- Extend to 3D or video segmentation
- Optimize using GPU or OpenMP
- Add automatic threshold tuning

---

## References

- Gonzalez & Woods – *Digital Image Processing*
- OpenCV Documentation: https://docs.opencv.org/
- Wikipedia – [Region Growing](https://en.wikipedia.org/wiki/Region_growing)

---

**Author**: Alina Macavei  
**Year**: 2025  



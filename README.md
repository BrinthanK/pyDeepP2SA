# pyDeepP2SA
## Code Documentation

pyDeepP2SA is an advanced particle size and shape analysis (P2SA) package that leverages the cutting-edge Segment Anything Model (SAM) developed by Facebook Inc. for highly accurate and robust object segmentation. Unlike traditional approaches that rely on manual training and old-fashioned watershed algorithms, pyDeepP2SA revolutionises the field by offering a zero-generalisation segmentation technique. With minimal manual intervention, this package delivers exceptional results and simplifies the entire analysis workflow.

### Prerequisites

To use pyDeepP2SA, you need to fulfill the following prerequisites:

- **Segment Anything Installation:** Install the "Segment Anything" package by running the following command:

  ```
  pip install git+https://github.com/facebookresearch/segment-anything.git
  ```

- **SAM Checkpoint:** Download one of the three available checkpoints provided by Facebook Inc. These checkpoints contain pre-trained models that are essential for the segmentation process. Save the downloaded checkpoint in a folder of your choice or on Google Drive if you are using the Colab platform. You can download the checkpoints from the [Segment Anything Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) repository.

Please ensure that you have installed the "Segment Anything" package and obtained the SAM checkpoint before proceeding with pyDeepP2SA.

### Installation 

Install pyDeepP2SA:

  ```
  pip install pyDeepP2SA
  ```

### Functions

#### `generate_masks(image, sam_checkpoint, points_per_side=32, pred_iou_thresh=0.95, stability_score_thresh=0.9, crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100)`

This function generates masks for an input image using a SAM checkpoint.

- `image`: The input image for which masks need to be generated.
- `sam_checkpoint`: The path to the SAM checkpoint file.
- `points_per_side`: The number of points per side used for generating the masks (default: 32).
- `pred_iou_thresh`: The predicted IoU (Intersection over Union) threshold used for generating the masks (default: 0.95).
- `stability_score_thresh`: The stability score threshold used for generating the masks (default: 0.9).
- `crop_n_layers`: The number of layers used for cropping during mask generation (default: 1).
- `crop_n_points_downscale_factor`: The downscale factor used for cropping during mask generation (default: 2).
- `min_mask_region_area`: The minimum region area for a mask to be considered valid (default: 100).

Returns:
- `masks`: A list of generated masks.

#### `visualise_masks(image, masks)`

This function visualises the segmented image with the generated masks.

- `image`: The input image.
- `masks`: The list of masks to visualise.

#### `save_masks_to_csv(masks, csv_directory, pixel_to_micron)`

This function saves the generated masks to a CSV file along with the calculated region properties.

- `masks`: The list of masks.
- `csv_directory`: The directory to save the CSV file.
- `pixel_to_micron`: The conversion factor from pixels to microns.

#### `plot_diameters(image, masks, diameter_threshold, circularity_threshold, pixel_to_micron)`

This function plots the original image with bounding boxes around the masks that have a diameter and circularity above the specified thresholds.

- `image`: The original input image.
- `masks`: The list of masks.
- `diameter_threshold`: The diameter threshold for filtering masks.
- `circularity_threshold`: The circularity threshold for filtering masks.
- `pixel_to_micron`: The conversion factor from pixels to microns, where "micron" is used as a unit name without any specific conversion factor associated with its general meaning.

#### `ind_mask(masks, diameter_threshold, circularity_threshold, pixel_to_micron, image)`

This function filters the masks based on diameter and circularity thresholds and plots the filtered masks along with their region properties.

- `masks`: The list of masks.
- `diameter_threshold`: The diameter threshold for filtering masks.
- `circularity_threshold`: The circularity threshold for filtering masks.
- `pixel_to_micron`: The conversion factor from pixels to microns.
- `image`: The original input image.

#### `stat_sum(diameter_threshold, circularity_threshold, csv_directory)`

This function reads the CSV file with region properties and returns a summary of the properties for the masks that meet the diameter and circularity thresholds.

- `diameter_threshold`: The diameter threshold for filtering masks.
- `circularity_threshold`: The circularity threshold for filtering masks.
- `csv_directory`: The directory where the CSV file is located.

Returns:
- `summary`: A summary of the filtered mask region properties.

#### `plot_boxplots(diameter_threshold, circularity_threshold, csv_directory)`

This function reads the CSV file and plots boxplots for the area, perimeter, diameter, and circularity of the masks that meet the diameter

 and circularity thresholds.

- `diameter_threshold`: The diameter threshold for filtering masks.
- `circularity_threshold`: The circularity threshold for filtering masks.
- `csv_directory`: The directory where the CSV file is located.

#### `plot_psd(diameter_threshold, circularity_threshold, csv_directory)`

This function reads the CSV file and plots the particle size distribution (PSD) using a histogram and cumulative frequency curve.

- `diameter_threshold`: The diameter threshold for filtering masks.
- `circularity_threshold`: The circularity threshold for filtering masks.
- `csv_directory`: The directory where the CSV file is located.

#### `plot_cir(diameter_threshold, circularity_threshold, csv_directory)`

This function reads the CSV file and plots the circularity distribution using a histogram.

- `diameter_threshold`: The diameter threshold for filtering masks.
- `circularity_threshold`: The circularity threshold for filtering masks.
- `csv_directory`: The directory where the CSV file is located.


#### `line_scan(image, image_bse, masks, circularity_threshold, min_area, csv_file, pixel_to_micron, line_distance_man, plot=False)``

Perform line scanning analysis on segmented masks within an image and store the results in a CSV file.

- `image`: The original image in which the masks are segmented.
- `image_bse`: A backscattered electron (BSE) image corresponding to the input image.
- `masks`: A list of dictionaries, where each dictionary represents a segmented mask and its properties. Each dictionary should have the following keys:
    - `segmentation`: A binary mask representing the segmented region.
    - `area`: The area of the segmented region in square pixels.
    - `predicted_iou`: Predicted intersection over union (IOU) value for the mask.
    - `bbox`: Bounding box coordinates [x, y, w, h] of the segmented region.
    - `point_coords`: Coordinates of points within the mask.
    - `stability_score`: A stability score for the mask.
    - `crop_box`: Bounding box coordinates [x, y, w, h] of the cropped mask region.
- `circularity_threshold`: Minimum circularity value for a mask to be considered in the analysis.
- `min_area`: Minimum area of a mask (in square micrometers) to be considered in the analysis.
- `csv_file`: Path to the CSV file where mask details and analysis results will be saved.
- `pixel_to_micron`: Conversion factor to convert pixel measurements to micrometers.
- `line_distance_man`: Manually set line scanning distance in pixels.
- `plot`: If True, generate and display plots during the analysis (default is False).

This function performs line scanning analysis on segmented masks to classify each mask as a 'cenosphere' or a 'solid sphere' based on the presence of line minima within the mask. The results, including circularity, area, perimeter, diameter, and type of each mask, are written to a CSV file.

Circular masks with circularity above the specified threshold and area greater than the specified minimum are considered for line scanning. For each mask, a line scanning analysis is performed along the vertical axis within the bounding box of the mask. The analysis includes fitting a polynomial curve to the pixel values along the line scan and identifying maxima and minima points.

The total number of line minima indices is used to classify the mask as a 'cenosphere' if it is greater than 0, or a 'solid sphere' if it is 0.

Region properties (area, perimeter, diameter) are also calculated and converted from pixels to micrometers using the provided conversion factor. These properties are added to the CSV file, along with the mask details.

### Dependencies

The code relies on the following dependencies:

- `numpy`: A library for numerical operations in Python.
- `scikit-image`: A library for image processing in Python.
- `torch` and `torchvision`: Libraries for deep learning and computer vision tasks.
- `opencv`: A library for image and video manipulation.
- `matplotlib`: A library for creating visualisations in Python.
- `pandas`: A library for data manipulation and analysis.
- `seaborn`: A library for statistical data visualisation.

Make sure these dependencies are installed in your Python environment before running the code.

### Usage

To use the provided functions, follow these steps:

1. Ensure that you have the required dependencies installed.

2. Import the required functions from the code file into your own Python script or interactive session.

3. Load an image of interest.

4. Call the `generate_masks` function, passing the image and SAM checkpoint file path as arguments. This will generate masks for the image.

5. Visualise the generated masks using the `visualise_masks` function, passing the original image and generated masks as arguments.

6. Save the generated masks and their region properties to a CSV file using the `save_masks_to_csv` function, providing the masks, the directory to save the CSV file, and the conversion factor from pixels to microns.

7. Optionally, plot the original image with bounding boxes around the masks that meet specific diameter and circularity thresholds using the `plot_diameters` function.

8. Further filter the masks based on diameter and circularity thresholds, and plot the filtered masks along with their region properties using the `ind_mask` function.

9. Generate a summary of the filtered mask region properties using the `stat_sum` function.

10. Plot boxplots of the area, perimeter, diameter, and circularity for the filtered masks using the `plot_boxplots` function.

11. Plot the particle size distribution (PSD) using a histogram and cumulative frequency curve using the `plot_psd` function.

12. Plot the circularity distribution using a histogram using the `plot_cir` function.

Please note that you may need to provide additional arguments or modify the existing ones based on your specific use case.

I hope this documentation helps you understand and utilise the provided code effectively! Let me know if you have any further questions.
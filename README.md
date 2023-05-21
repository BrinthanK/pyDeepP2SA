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

This function visualizes the segmented image with the generated masks.

- `image`: The input image.
- `masks`: The list of masks to visualize.

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
- `pixel_to_micron`: The conversion factor from pixels to microns.

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


### Dependencies

The code relies on the following dependencies:

- `numpy`: A library for numerical operations in Python.
- `scikit-image`: A library for image processing in Python.
- `torch` and `torchvision`: Libraries for deep learning and computer vision tasks.
- `opencv`: A library for image and video manipulation.
- `matplotlib`: A library for creating visualizations in Python.
- `pandas`: A library for data manipulation and analysis.
- `seaborn`: A library for statistical data visualization.

Make sure these dependencies are installed in your Python environment before running the code.

### Usage

To use the provided functions, follow these steps:

1. Ensure that you have the required dependencies installed.

2. Import the required functions from the code file into your own Python script or interactive session.

3. Load an image of interest.

4. Call the `generate_masks` function, passing the image and SAM checkpoint file path as arguments. This will generate masks for the image.

5. Visualize the generated masks using the `visualise_masks` function, passing the original image and generated masks as arguments.

6. Save the generated masks and their region properties to a CSV file using the `save_masks_to_csv` function, providing the masks, the directory to save the CSV file, and the conversion factor from pixels to microns.

7. Optionally, plot the original image with bounding boxes around the masks that meet specific diameter and circularity thresholds using the `plot_diameters` function.

8. Further filter the masks based on diameter and circularity thresholds, and plot the filtered masks along with their region properties using the `ind_mask` function.

9. Generate a summary of the filtered mask region properties using the `stat_sum` function.

10. Plot boxplots of the area, perimeter, diameter, and circularity for the filtered masks using the `plot_boxplots` function.

11. Plot the particle size distribution (PSD) using a histogram and cumulative frequency curve using the `plot_psd` function.

12. Plot the circularity distribution using a histogram using the `plot_cir` function.

Please note that you may need to provide additional arguments or modify the existing ones based on your specific use case.

I hope this documentation helps you understand and utilize the provided code effectively! Let me know if you have any further questions.
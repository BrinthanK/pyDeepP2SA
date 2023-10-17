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


#### `line_scan(image, image_bse, masks, circularity_threshold, min_area, csv_file, pixel_to_micron, line_distance_man, plot=False)`

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


Certainly! Here's the documentation for the provided functions in Markdown format:

#### `plot_segment_bounding_boxes(csv_file, segment_types, image)`

This function plots bounding boxes on an image based on mask details from a CSV file for specific segment types.

- **csv_file** (str): The path to the CSV file containing mask details.
- **segment_types** (list): A list of segment types (e.g., ['cenosphere', 'solid sphere']) for which bounding boxes will be plotted.
- **image** (ndarray): The original image on which the bounding boxes will be overlaid.

The function reads the CSV file, filters mask details based on the specified segment types, and then creates a plot with bounding boxes. Each bounding box corresponds to a mask in the CSV file and is color-coded based on the segment type. A legend is included to explain the color-coding. The resulting plot is displayed.

#### `psd_spheres(csv_file)`

This function plots a particle size distribution (PSD) for cenospheres and solid spheres based on diameter information from a CSV file.

- **csv_file** (str): The path to the CSV file containing mask details, including diameter information.

The function reads the CSV file, separates masks into cenospheres and solid spheres, and then creates a histogram with kernel density estimation (KDE) for each category. The cenospheres are displayed in a color coded as '#FFBE86', and solid spheres are displayed in a color coded as '#8EBAD9'. The x-axis represents the diameter in micrometers, and the y-axis represents the count of particles. A legend is included to distinguish between cenospheres and solid spheres.

#### `box_plots_spheres(csv_file)`

This function creates box plots to compare the diameters of cenospheres and solid spheres based on mask details from a CSV file.

- **csv_file** (str): The path to the CSV file containing mask details, including diameter information.

The function reads the CSV file, separates masks into cenospheres and solid spheres, and then generates a box plot to compare the diameters of the two categories. The x-axis is labeled with 'Cenospheres' and 'Solid Spheres,' while the y-axis represents the diameter in micrometers. The title of the plot is 'Box Plot - Cenospheres vs Solid Spheres.'

These functions are designed for visualizing and analyzing mask details and particle size distributions based on the information stored in the CSV file. You can use them to gain insights into the characteristics of segmented objects in your images.


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

Please note that you may need to provide additional arguments or modify the existing ones based on your specific use case.

I hope this documentation helps you understand and utilise the provided code effectively! Let me know if you have any further questions.
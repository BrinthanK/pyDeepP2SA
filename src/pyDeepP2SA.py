import csv
import numpy as np
from skimage import measure
import torch
import torchvision
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from skimage.segmentation import clear_border
import math
from scipy.signal import argrelextrema
from skimage import measure
from scipy.optimize import curve_fit
import matplotlib.patches as patches

# Particle shape and size analysis section 
def generate_masks(image, sam_checkpoint,
                   points_per_side=32, pred_iou_thresh=0.95, stability_score_thresh=0.9,
                   crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100):
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator_ = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )

    masks = mask_generator_.generate(image)
    return masks

def visualise_masks(image, masks):
    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.35)))

    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.axis('off')
    plt.title("Segmented image with " + str(len(masks)) + " segments" )
    plt.show()

def save_masks_to_csv(masks, csv_directory, pixel_to_micron):
    with open(csv_directory, 'w', newline='') as csvfile:
        fieldnames = ['area', 'perimeter', 'diameter', 'circularity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for mask in masks:
            # Extract the segmentation from the mask
            segmentation = mask['segmentation']

            # Create a labeled mask using the segmentation and the 'true' value
            labeled_mask = np.zeros_like(segmentation, dtype=np.uint8)
            labeled_mask[segmentation] = 1
            labeled_mask = measure.label(labeled_mask)

            # Remove segments touching the border
            cleared_mask = clear_border(labeled_mask)

            # Loop over each connected component and extract its region props
            for region in measure.regionprops(cleared_mask):
                # Convert area, perimeter, and diameter from pixels to micrometers
                area_pixels = region.area
                perimeter_pixels = region.perimeter
                diameter_pixels = region.major_axis_length

                # Convert measurements to microns
                area = area_pixels * pixel_to_micron**2
                perimeter = perimeter_pixels * pixel_to_micron
                diameter = diameter_pixels * pixel_to_micron

                # Calculate circularity as 4π(area/perimeter^2)
                circularity = (4 * np.pi * area) / (perimeter ** 2)

                # Write region props to CSV file
                writer.writerow({'area': area, 'perimeter': perimeter,
                                 'diameter': diameter, 'circularity': circularity})
                
def plot_diameters(image, masks, diameter_threshold, circularity_threshold, pixel_to_micron):
    fig, ax = plt.subplots()

    for mask in masks:
        # Extract the segmentation from the mask
        segmentation = mask['segmentation']

        # Create a labeled mask using the segmentation and the 'true' value
        labeled_mask = np.zeros_like(segmentation, dtype=np.uint8)
        labeled_mask[segmentation] = 1
        labeled_mask = measure.label(labeled_mask)
        labeled_mask = clear_border(labeled_mask)

        # Loop over each connected component and extract its region props
        for region in measure.regionprops(labeled_mask):
            # Get the bounding box coordinates
            minr, minc, maxr, maxc = region.bbox

            # Calculate the diameter
            diameter = region.major_axis_length* pixel_to_micron
            circularity = (4 * region.area * np.pi) / (region.perimeter ** 2)

            # Filter based on the diameter and circularity thresholds
            if diameter > diameter_threshold and circularity > circularity_threshold:
                # Plot the segmentation with the bounding box
                ax.imshow(image, cmap='gray')
                rect = Rectangle((minc, minr), maxc - minc, maxr - minr,
                                 fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)

                # Add the diameter as text
                #ax.text(minc, minr, f'D: {diameter:.2f}', color='red',
                        #verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})

    ax.axis('off')
    plt.show()

def ind_mask(masks, diameter_threshold, circularity_threshold, pixel_to_micron, image):
    filtered_masks = []

    for mask in masks:
        # Extract the segmentation from the mask
        segmentation = mask['segmentation']

        # Create a labeled mask using the segmentation and the 'true' value
        labeled_mask = np.zeros_like(segmentation, dtype=np.uint8)
        labeled_mask[segmentation] = 1
        labeled_mask = measure.label(labeled_mask)

        # Remove segments touching the border
        labeled_mask = clear_border(labeled_mask)

        # Loop over each connected component and extract its region props
        for region in measure.regionprops(labeled_mask):
            # Calculate the diameter
            diameter = region.major_axis_length * pixel_to_micron

            # Calculate the circularity
            circularity = (4 * region.area * np.pi) / (region.perimeter ** 2)

            # Filter based on the diameter and circularity thresholds
            if diameter > diameter_threshold and circularity > circularity_threshold:
                filtered_masks.append((mask, region))
                break  # Found a satisfying mask, no need to process further masks in the loop

    num_plots = len(filtered_masks)
    num_cols = min(3, num_plots)  # Set a maximum of 3 columns
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, (mask, region) in enumerate(filtered_masks):
        # Extract the segmentation from the mask
        segmentation = mask['segmentation']

        # Create a labeled mask using the segmentation and the 'true' value
        labeled_mask = np.zeros_like(segmentation, dtype=np.uint8)
        labeled_mask[segmentation] = 1
        labeled_mask = measure.label(labeled_mask)

        # Remove segments touching the border
        labeled_mask = clear_border(labeled_mask)

        # Create a subplot for the current mask
        ax = axes[i]

        # Show the image
        ax.axis('off')

        ax.imshow(image)

        # Loop over each connected component and extract its region props
        for region in measure.regionprops(labeled_mask):
            # Calculate the diameter
            diameter = region.major_axis_length * pixel_to_micron

            # Calculate the circularity
            circularity = (4 * region.area * np.pi) / (region.perimeter ** 2)

            # Draw a bounding box around the region
            min_row, min_col, max_row, max_col = region.bbox
            rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                 fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

            # Create the caption text
            caption = f"Area: {region.area * pixel_to_micron * pixel_to_micron:.2f}\nPerimeter: {region.perimeter * pixel_to_micron:.2f}\nDiameter: {diameter:.2f}\nCircularity: {circularity:.2f}"

            # Add the caption as title to the subplot
            ax.set_title(caption, color='red', fontsize=10)

    # Hide empty subplots
    for i in range(len(filtered_masks), len(axes)):
        axes[i].axis('off')

    # Adjust subplot spacing
    fig.tight_layout()

    # Show the plot
    plt.show()


def stat_sum(diameter_threshold, circularity_threshold,csv_directory):
    stat = pd.read_csv(csv_directory)

    filtered_stat = stat[(stat['diameter'] > diameter_threshold) & (stat['circularity'] > circularity_threshold)]

    summary = filtered_stat.describe()

    return summary

def plot_boxplots(diameter_threshold,circularity_threshold,csv_directory):
    stat = pd.read_csv(csv_directory)
    stat = stat[(stat['diameter'] > diameter_threshold) & (stat['circularity'] > circularity_threshold)]
    data = [stat["area"].dropna(), stat["perimeter"].dropna(), stat["diameter"].dropna(),
        stat["circularity"].dropna()]

    data_names = ["Area", "Perimeter", "Diameter", "Circularity"]


    # Create a figure and axes
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # Loop over the axes and create a box plot for each dataset
    for i, ax in enumerate(axs):
        ax.boxplot(data[i])

        # Set the title and y-axis label for each subplot
        ax.set_title(data_names[i])
        ax.set_ylabel("Value")

        # Remove x-axis labels and ticks
        ax.set_xticks([])
        ax.set_xticklabels([])

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_psd(diameter_threshold, circularity_threshold, num_bins,csv_directory):
    stat = pd.read_csv(csv_directory)

    # Apply diameter and circularity thresholds
    filtered_stat = stat[(stat['diameter'] > diameter_threshold) & (stat['circularity'] > circularity_threshold)]

    f, ax = plt.subplots()
    ax1 = sns.histplot(data=filtered_stat["diameter"], bins=num_bins, kde=True,
                       line_kws={"linewidth": 2, 'color': '#b44a46'})
    ax1 = plt.twinx()
    sns.ecdfplot(data=filtered_stat, x="diameter", ax=ax1, stat="proportion", color='red',
                 linewidth=2)

    ax.set(xlabel='Particle size', ylabel='Number of particles')
    ax1.set_ylabel('Cumulative frequency of particles')
    plt.xticks(fontsize=12)
    plt.title("Particle size distribution")
    plt.show()

def plot_cir(diameter_threshold, circularity_threshold, num_bins, csv_directory):
    stat = pd.read_csv(csv_directory)

    # Apply diameter and circularity thresholds
    filtered_stat = stat[(stat['diameter'] > diameter_threshold) & (stat['circularity'] > circularity_threshold)]

    f, ax = plt.subplots()
    ax1 = sns.histplot(data=filtered_stat["circularity"], bins=num_bins, kde=True,
                       line_kws={"linewidth": 2, 'color': '#b44a46'})
    

    ax.set(xlabel='Circularity', ylabel='Number of particles')
    plt.xticks(fontsize=12)
    plt.title("Circularity distribution")
    plt.show()

# Characterisation section 

def line_scan(image, image_bse, masks, circularity_threshold, min_area, csv_file, pixel_to_micron, line_distance_man, plot=False):
    # Resize the BSE image to match the input image
    image_bse = resize(image_bse, image.shape)

    mask_details = []

    # Open the CSV file for writing mask details
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['Mask', 'Area', 'Circularity', 'Type', 'BBox', 'Predicted IOU', 'Point Coords',
                      'Stability Score', 'Perimeter', 'Diameter']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the CSV header
        writer.writeheader()

        cenosphere_images = []
        solidsphere_images = []

        # Process each mask
        for i, mask in enumerate(masks):
            segmentation_array = mask['segmentation']
            area = mask['area']
            predicted_iou = mask['predicted_iou']
            bbox = mask['bbox']  # Bounding box [x, y, w, h]
            point_coords = mask['point_coords']
            stability_score = mask['stability_score']
            crop_box = mask['crop_box']

            # Crop the segmented image using the bounding box
            x, y, w, h = bbox
            segmented_image = cv2.bitwise_and(image_bse, image_bse, mask=segmentation_array.astype(np.uint8))
            cropped_segmented_image = segmented_image[y:y+h, x:x+w]

            # Function to calculate circularity of a mask
            def calculate_circularity(segmentation_array):
                contours, _ = cv2.findContours(segmentation_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                perimeter = cv2.arcLength(contours[0], True)
                area = cv2.contourArea(contours[0])
                if perimeter == 0:
                    return 0
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                return circularity

            # Function for the polynomial fit
            def polynomial_func(x, a, b, c, d, e):
                return a * x**4 + b * x**3 + c * x**2 + d * x + e

            # Calculate circularity
            circularity = calculate_circularity(segmentation_array)

            # Check circularity and area thresholds
            if circularity > circularity_threshold and area > min_area:
                height = cropped_segmented_image.shape[0]
                start_line = int(0.1 * height)  # Start line at 10% of the height
                end_line = int(0.9 * height)  # End line at 90% of the height
                line_distance = end_line - start_line

                num_line_scans = int(line_distance / (line_distance_man / pixel_to_micron))
                line_scan_indices = np.linspace(start_line, end_line, num_line_scans, dtype=int)
                line_scans = []  # Store line scan data for each mask

                # Perform line scanning
                for line_index in line_scan_indices:
                    line_pixel_values = cropped_segmented_image[line_index, :].flatten()
                    x = np.arange(len(line_pixel_values))

                    # Fit a polynomial curve to the line scan data
                    popt_line, pcov_line = curve_fit(polynomial_func, x, line_pixel_values)
                    fitted_curve_line = polynomial_func(x, *popt_line)

                    # Find the maxima or minima points of the fitted curve
                    line_extremum_index = np.argmax(fitted_curve_line) if popt_line[0] < 0 else np.argmin(fitted_curve_line)
                    line_extremum_x = x[line_extremum_index]
                    line_extremum_y = fitted_curve_line[line_extremum_index]

                    line_maxima_indices = argrelextrema(fitted_curve_line, np.greater)[0]
                    line_minima_indices = argrelextrema(fitted_curve_line, np.less)[0]

                    # Store line scan data
                    line_scans.append({
                        'line_pixel_values': line_pixel_values,
                        'fitted_curve_line': fitted_curve_line,
                        'line_extremum_x': line_extremum_x,
                        'line_extremum_y': line_extremum_y,
                        'line_maxima_indices': line_maxima_indices,
                        'line_minima_indices': line_minima_indices
                    })

                # Count the total number of line_minima_indices
                total_line_minima_indices = sum(len(scan['line_minima_indices']) for scan in line_scans)

                # Classify the segment
                segment_type = 'cenosphere' if total_line_minima_indices > 0 else 'solid sphere'

                # Save additional region properties
                labeled_mask = np.zeros_like(segmentation_array, dtype=np.uint8)
                labeled_mask[segmentation_array] = 1
                labeled_mask = measure.label(labeled_mask)
                cleared_mask = clear_border(labeled_mask)
                region_props = measure.regionprops(cleared_mask)

                for region in region_props:
                    # Convert area, perimeter, and diameter from pixels to micrometers
                    area_pixels = region.area
                    perimeter_pixels = region.perimeter
                    diameter_pixels = region.major_axis_length

                    # Convert measurements to microns
                    area_2 = area_pixels * pixel_to_micron**2
                    perimeter = perimeter_pixels * pixel_to_micron
                    diameter = diameter_pixels * pixel_to_micron

                    # Append mask details to the list
                    mask_details.append({
                        'Mask': i + 1,
                        'Area': area * pixel_to_micron**2,
                        'Circularity': circularity,
                        'BBox': bbox,
                        'Predicted IOU': predicted_iou,
                        'Point Coords': point_coords,
                        'Stability Score': stability_score,
                        'Perimeter': perimeter,
                        'Diameter': diameter,
                        'Type': segment_type
                    })

                    # Write mask details to CSV file
                    writer.writerow({
                        'Mask': i + 1, 'Area': area, 'Circularity': circularity, 'BBox': bbox,
                        'Predicted IOU': predicted_iou, 'Point Coords': point_coords,
                        'Stability Score': stability_score, 'Perimeter': perimeter,
                        'Diameter': diameter, 'Type': segment_type
                    })

def plot_segment_bounding_boxes(csv_file, segment_types, image):
    # Read the CSV file and extract the relevant data
    mask_details = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mask_details.append(row)

    # Filter the mask details based on the segment types
    filtered_mask_details = [mask for mask in mask_details if mask['Type'] in segment_types]

    # Create a single plot
    fig, ax = plt.subplots()

    # Load and plot the corresponding images with bounding boxes
    colors = ['r', 'b']  # Colors for each segment type

    for idx, segment_type in enumerate(segment_types):
        segments = [mask for mask in filtered_mask_details if mask['Type'] == segment_type]

        for mask in segments:
            bbox = eval(mask['BBox'])

            # Plot the image
            plt.imshow(image, cmap='gray')

            # Create a rectangle patch for the bounding box
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=colors[idx], facecolor='none')

            # Add the rectangle patch to the plot
            ax.add_patch(rect)

    # Set the title
    #plt.title(f"Segment Types: {', '.join(segment_types)}")

    # Create custom legend patches
    legend_patches = [patches.Patch(facecolor='none', edgecolor=colors[i], label=segment_types[i]) for i in range(len(segment_types))]

    # Add the legend with the custom patches
    plt.legend(handles=legend_patches,bbox_to_anchor=(0.5, -0.02), loc='upper center', ncol=len(segment_types))

    plt.axis('off')
    plt.tight_layout()

    # Show the plot
    plt.show()

def psd_spheres(csv_file):
    cenosphere_sizes = []
    solid_sizes = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mask_type = row['Type']
            diameter = float(row['Diameter'])

            if mask_type == 'cenosphere':
                cenosphere_sizes.append(diameter)
            else:
                solid_sizes.append(diameter)

    fig, ax = plt.subplots()
    sns.histplot(cenosphere_sizes, bins=10, kde=True, color='#FFBE86', label='Cenospheres', ax=ax)
    sns.histplot(solid_sizes, bins=10, kde=True, color='#8EBAD9', label='Solid Spheres', ax=ax)
    ax.set_xlabel('Diameter (µm)')
    ax.set_ylabel('Count')
    ax.legend()
    plt.show()

def box_plots_spheres(csv_file):
    cenosphere_sizes = []
    solid_sizes = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mask_type = row['Type']
            diameter = float(row['Diameter'])

            if mask_type == 'cenosphere':
                cenosphere_sizes.append(diameter)
            else:
                solid_sizes.append(diameter)

    fig, ax = plt.subplots()
    sns.boxplot(data=[cenosphere_sizes, solid_sizes], ax=ax)
    ax.set_xticklabels(['Cenospheres', 'Solid Spheres'])
    ax.set_ylabel('Diameter (µm)')
    ax.set_title('Box Plot - Cenospheres vs Solid Spheres')
    plt.show()

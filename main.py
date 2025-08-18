from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pydicom
from PIL import Image
import sys

def tinting(input_path, output_path):
    # Load BGR image as float32 for calculation
    image = cv2.imread(input_path).astype(np.float32)

    # Split channels
    b, g, r = cv2.split(image)

    # Create mask where all r, g, and b are >= 5
    mask = (r >= 5) & (g >= 5) & (b >= 5)

    # Apply tinting only where mask is True
    r_tinted = 0.7944 * r + 24.8147
    g_tinted = 0.8392 * g + 26.4082
    b_tinted = 0.9571 * b + 33.2031

    # Combine tinted and original values based on mask
    r = np.where(mask, r_tinted, r)
    g = np.where(mask, g_tinted, g)
    b = np.where(mask, b_tinted, b)

    # Clip values to [0, 255]
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    # Convert back to uint8 and merge channels
    processed = cv2.merge([b, g, r]).astype(np.uint8)

    # Save result
    cv2.imwrite(output_path, processed)

def get_gmm_thresholds(pixel_values, n_components=3):
    pixel_values = pixel_values.reshape(-1, 1).astype(np.float32)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(pixel_values)

    # Get the means and sort them to assign to tissue classes
    means = gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    sorted_means = means[sorted_idx]

    # Midpoints between means = thresholds
    T1 = int((sorted_means[0] + sorted_means[1]) / 2)  # muscle and fat boundary
    T2 = int((sorted_means[1] + sorted_means[2]) / 2)  # fat and fibrous boundary

    return T1, T2

def validate_or_manual(img, mask):
    # Make model inference overlay bluish instead of white
    blue_mask = cv2.applyColorMap(mask, cv2.COLORMAP_OCEAN)
    overlay = cv2.addWeighted(img, 0.7, blue_mask, 0.3, 0)
    cv2.imshow('Model Inference', overlay)
    print('Press "y" to accept model mask, "n" to decline and manually segment')
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord('y'):
        print('Accepted model mask')
        return mask

    # cdlined
    print('Manual Segmentation Selected')
    manual_mask = np.zeros(img.shape[:2], np.uint8)
    clone = img.copy()
    points = []
    first_point = None
    closed = False
    tolerance = 10

    def draw_polygon(event, x, y, flags, param):
        nonlocal first_point, closed
        if closed:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if not points:  # initial
                first_point = (x, y)
                points.append((x, y))
                cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)
            else:
                # Check near
                if abs(x - first_point[0]) < tolerance and abs(y - first_point[1]) < tolerance:
                    points.append(first_point)  # close polygon
                    cv2.polylines(clone, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.fillPoly(manual_mask, [np.array(points)], 255)
                    cv2.imshow('Manual Segmentation', clone)
                    print('Saved manual segmentation')
                    closed = True
                else:
                    points.append((x, y))
                    cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)
                    cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow('Manual Segmentation', clone)

    cv2.imshow('Manual Segmentation', clone)
    cv2.setMouseCallback('Manual Segmentation', draw_polygon)
    while not closed:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to abort
            break
    cv2.destroyAllWindows()
    return manual_mask


def main():
    # Loadings; please note enter the file and remove the suffix. File must be in the same directory as main.py
    name='IM-0003-0030'

    # Conv to png
    ds = pydicom.dcmread(f'{name}.dcm') # path
    new_image = ds.pixel_array.astype(float)

    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0

    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)

    # final_image.show() # Display the Image (test purposes)

    final_image.save(f'{name}.png') # Save the image as PNG

    # more loading
    model = YOLO('best.pt')
    tinting(f'{name}.png', f'{name}-adj.png')
    img = cv2.imread(f'{name}-adj.png')
    try:
        results = model(img, verbose=False)[0]
    except TypeError:
        print('no rectus femoris detected')
        sys.exit()

    # --- Get mask with highest confidence ---
    if results.masks is not None and results.boxes is not None:
        confidences = results.boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)  # index of highest confidence object

        # Extract only that mask
        best_mask = results.masks.data[best_idx].cpu().numpy().astype(np.uint8) * 255
        mask_total = cv2.resize(best_mask, (img.shape[1], img.shape[0]))
    else:
        print('No rectus femoris detected')
        sys.exit()
    img = cv2.imread(f'{name}.png')
    mask_total = validate_or_manual(img, mask_total)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask_total  # masking

    cv2.imwrite(f'{name}_segmentation_mask.png', rgba)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask_total)

    # --- Triple-Threshold Segmentation ---
    # Compute T1, T2 using GMM
    rf_pixels = masked_gray[mask_total > 0].flatten()
    T1, T2 = get_gmm_thresholds(rf_pixels)
    print(f'GMM thresholds: T1: {T1}, T2: {T2}')


    # Calculate areas
    muscle_area = np.count_nonzero((masked_gray < T1) & (mask_total > 0))
    fat_area = np.count_nonzero((masked_gray >= T1) & (masked_gray < T2) & (mask_total > 0))
    fibrous_area = np.count_nonzero((masked_gray >= T2) & (mask_total > 0))
    total_area = np.count_nonzero(mask_total)

    # Percentages
    muscle_pct = (muscle_area / total_area) * 100
    fat_pct = (fat_area / total_area) * 100
    fibrous_pct = (fibrous_area / total_area) * 100

    print(f'Total object area: {total_area} pixels')
    print(f'Muscle content (<{T1}): {muscle_pct:.2f}%')
    print(f'Fat content ({T1}-{T2}): {fat_pct:.2f}%')
    print(f'Fibrous content (>{T2}): {fibrous_pct:.2f}%')

    # Highlight masks
    highlight = img.copy()
    muscle_mask = (masked_gray < T1) & (mask_total > 0)
    fat_mask = (masked_gray >= T1) & (masked_gray < T2) & (mask_total > 0)
    fibrous_mask = (masked_gray >= T2) & (mask_total > 0)

    highlight[muscle_mask] = [0, 0, 255]     # Blue for muscle
    highlight[fat_mask] = [0, 255, 255]        # Green for fat
    highlight[fibrous_mask] = [0, 255, 0]    # Red for fibrous

    cv2.imwrite(f'{name}_highlighted_output.png', highlight)
    highlight_rgba = cv2.cvtColor(highlight, cv2.COLOR_BGR2BGRA)
    highlight_rgba[:, :, 3] = mask_total
    
    cv2.imwrite(f'{name}_highlighted_transparent.png', highlight_rgba)
    # Flatten grayscale values inside the mask
    rf_pixels = masked_gray[mask_total > 0].flatten()

    # --- Save individual component highlights ---
    # Start from the original image
    muscle_only = img.copy()
    fat_only = img.copy()
    fibrous_only = img.copy()

    # Zero out pixels that are not part of the target tissue
    muscle_only[~muscle_mask] = img[~muscle_mask]
    fat_only[~fat_mask] = img[~fat_mask]
    fibrous_only[~fibrous_mask] = img[~fibrous_mask]

    # Color only the target tissue in place
    muscle_only[muscle_mask] = [0, 0, 255]     # Blue
    fat_only[fat_mask] = [0, 255, 255]         # Yellow-green
    fibrous_only[fibrous_mask] = [0, 255, 0]   # Green

    # Save the images
    cv2.imwrite(f'{name}_highlighted_muscle_only.png', muscle_only)
    cv2.imwrite(f'{name}_highlighted_fat_only.png', fat_only)
    cv2.imwrite(f'{name}_highlighted_fibrous_only.png', fibrous_only)



    # Plot histogram
    plt.figure(figsize=(10, 4))
    plt.hist(rf_pixels, bins=100, color='gray', edgecolor='black')
    plt.axvline(T1, color='blue', linestyle='--', label=f'T1 = {T1} (muscle↔fat)')
    plt.axvline(T2, color='red', linestyle='--', label=f'T2 = {T2} (fat↔fibrous)')
    plt.title(f'{name}: Pixel Intensity Histogram within Rectus Femoris (W/GMM)')
    plt.xlabel('Grayscale Intensity (0-255)')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{name}_grayscale_histogram.png')
    plt.close()


if __name__ == '__main__':
    main()
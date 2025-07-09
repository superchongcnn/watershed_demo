import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def watershed_segmentation_with_example():
    # --- CHANGE THIS TO YOUR ACTUAL IMAGE PATH ---
    # For demonstration, I'll download a sample image.
    # If you have 'ct.jpg' with objects, use that path directly.
    
    # Try to download a sample image if not present (e.g., for testing)
    sample_image_url = 'https://docs.opencv.org/4.x/coins.jpg'
    sample_image_name = 'coins.jpg' # Or 'ct.jpg' if you have it
    
    # Check if the sample image exists, if not, try to download it
    if not os.path.exists(sample_image_name):
        print(f"'{sample_image_name}' not found. Attempting to download for demonstration...")
        try:
            import requests
            response = requests.get(sample_image_url)
            response.raise_for_status() # Raise an exception for HTTP errors
            with open(sample_image_name, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded '{sample_image_name}' successfully.")
        except Exception as e:
            print(f"Could not download sample image: {e}")
            print("Please ensure you have an image with objects (e.g., 'ct.jpg' or 'coins.jpg') in the same directory.")
            return

    imgPath = os.path.join(sample_image_name) # Use the sample image or your CT scan image
    # imgPath = os.path.join('ct.jpg') # Uncomment and use this if you have your CT image

    # 1. **首先读取图像**
    img_original = cv.imread(imgPath) 
    
    # **关键检查：确保图片成功加载**
    if img_original is None:
        print(f"错误：无法加载图像。请检查路径是否正确：{imgPath}")
        print("请确认图片文件是否存在于脚本运行的目录中，或路径是否正确。")
        return

    # 2. **然后对读取到的图像进行裁剪 (Optional, adjust as needed)**
    # For 'coins.jpg' or general images, you might not need cropping, or adjust the range.
    # If your CT scan is large, keep the cropping.
    # The original image you uploaded is 600x900. Your crop [10:700, 10:900] for a 600x900 image
    # would result in a height of 690 and width of 890, which is larger than the original image's height.
    # Let's adjust for the provided image dimensions (600 height, 900 width) or general case.
    
    # If you are using your provided image_1c7adb.png, this crop needs to be within [0:600, 0:900]
    # For 'coins.jpg', no specific crop is needed for demonstration, so I'll comment it out.
    # If your 'ct.jpg' needs cropping, uncomment and adjust:
    # img_cropped = img_original[10:img_original.shape[0]-10, 10:img_original.shape[1]-10] 
    img_cropped = img_original # For coins.jpg, use the full image

    # Check for valid cropped image (if cropping is applied)
    if img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
        print("错误：裁剪后的图像为空。请检查裁剪范围是否超出了原始图像尺寸。")
        print(f"原始图像尺寸：{img_original.shape}")
        return

    # Convert to RGB for Matplotlib display
    imgRGB = cv.cvtColor(img_cropped, cv.COLOR_BGR2RGB)
    # Convert to grayscale for watershed processing
    img_gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)

    # --- Watershed Pre-processing Steps ---
    # The choice of threshold, kernel size, and distance transform threshold are critical
    # and highly dependent on the image content.

    # Otsu's thresholding is often better for automatically finding a good threshold for binary images.
    # It assumes a bimodal distribution in the histogram.
    ret, imgThreshold = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # If objects are dark on a light background, use THRESH_BINARY_INV.
    # If objects are light on a dark background, use THRESH_BINARY.
    # For coins.jpg, THRESH_BINARY_INV is good as coins are light and background dark after invert.
    print(f"Otsu's Threshold value: {ret}") # Print the determined threshold

    # Noise removal (optional, but good for real images)
    kernel_opening = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(imgThreshold, cv.MORPH_OPEN, kernel_opening, iterations = 2)

    # Sure background area
    sure_bg = cv.dilate(opening, kernel_opening, iterations=3)

    # Finding sure foreground area (markers)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    
    # A higher value here means only very "sure" foreground pixels are marked.
    # This threshold is crucial! Adjust based on your image.
    # For coins, a value like 0.7*max_dist_transform might work well.
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Finding unknown region
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the unknown region with 0
    markers[unknown == 255] = 0

    # --- Plotting Intermediate Results ---
    plt.figure(figsize=(18, 12)) 
    
    plt.subplot(241)
    plt.imshow(img_gray, cmap='gray')
    plt.title('1. Cropped Gray Image')

    plt.subplot(242)
    plt.imshow(imgThreshold, cmap='gray')
    plt.title(f'2. Binary Threshold (Otsu: {int(ret)})')

    plt.subplot(243)
    plt.imshow(opening, cmap='gray')
    plt.title('3. Morphological Opening (Noise Removal)')

    plt.subplot(244)
    plt.imshow(sure_bg, cmap='gray')
    plt.title('4. Sure Background')

    plt.subplot(245)
    plt.imshow(dist_transform, cmap='jet')
    plt.title('5. Distance Transform')

    plt.subplot(246)
    plt.imshow(sure_fg, cmap='gray')
    plt.title('6. Sure Foreground (Markers)')

    plt.subplot(247)
    plt.imshow(unknown, cmap='gray')
    plt.title('7. Unknown Region')

    plt.subplot(248)
    plt.imshow(markers, cmap='nipy_spectral')
    plt.title('8. Final Markers for Watershed')
    plt.tight_layout()
    
    # --- Watershed Algorithm ---
    # The watershed function modifies the 'markers' array in-place.
    # It takes a 3-channel color image and the markers.
    
    # Make a copy of the color image to draw on
    img_watershed_display = imgRGB.copy()
    
    # Apply watershed
    # Crucially, the markers array must be int32 type for cv.watershed
    markers = np.int32(markers) 
    labels_watershed = cv.watershed(img_watershed_display, markers)

    # --- Plotting Final Results ---
    plt.figure(figsize=(14, 7))
    
    plt.subplot(121)
    plt.imshow(labels_watershed, cmap='nipy_spectral')
    plt.title('Watershed Segmentation Result (Labels)')

    plt.subplot(122)
    # Mark the watershed boundaries in red on the original image
    img_watershed_display[labels_watershed == -1] = [255, 0, 0] # Boundaries are -1
    plt.imshow(img_watershed_display)
    plt.title('Original Image with Watershed Lines')
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    watershed_segmentation_with_example()
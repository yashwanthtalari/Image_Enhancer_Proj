import cv2
import numpy as np
import os

# -------------------- CONFIG --------------------
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- FUNCTIONS --------------------

def enhance_image(image):
    """
    Improves clarity, sharpness, and contrast
    """

    # 1. Denoise
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, 10, 10, 7, 21
    )

    # 2. Sharpen (Unsharp Mask)
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    # 3. Improve Contrast using CLAHE
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    merged = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return enhanced


def apply_filters(image):
    """
    Returns dictionary of filtered images
    """

    filters = {}

    # Grayscale
    filters["grayscale"] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge Detection
    filters["edges"] = cv2.Canny(image, 100, 200)

    # Gaussian Blur
    filters["blur"] = cv2.GaussianBlur(image, (7, 7), 0)

    # Bilateral Filter (smooth but keep edges)
    filters["bilateral"] = cv2.bilateralFilter(image, 9, 75, 75)

    return filters


# -------------------- MAIN LOOP --------------------

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(input_path)

        if image is None:
            continue

        # Enhance image
        enhanced = enhance_image(image)

        # Save enhanced image
        base, ext = os.path.splitext(filename)
        enhanced_path = os.path.join(OUTPUT_DIR, f"{base}_enhanced{ext}")
        cv2.imwrite(enhanced_path, enhanced)

        # Apply filters
        filtered_images = apply_filters(enhanced)

        for name, img in filtered_images.items():
            filter_path = os.path.join(
                OUTPUT_DIR, f"{base}_{name}{ext}"
            )

            # Handle grayscale separately
            if len(img.shape) == 2:
                cv2.imwrite(filter_path, img)
            else:
                cv2.imwrite(filter_path, img)

print("Processing complete. Enhanced images saved to output_images/")

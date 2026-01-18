import cv2  # type: ignore
import numpy as np
import os

# -------------------- CONFIG --------------------
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
MODEL_PATH = "ESPCN_x4.pb"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- ENHANCEMENT --------------------


def enhance_image(image):  # type: ignore
    """Enhance image with denoising, sharpening, and contrast enhancement."""
    try:
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
        )

        # Strong sharpening (Unsharp Mask)
        blur = cv2.GaussianBlur(denoised, (0, 0), 1.2)
        sharp = cv2.addWeighted(denoised, 1.8, blur, -0.8, 0)

        # High-boost sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(sharp, -1, kernel)

        # CLAHE contrast enhancement
        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Micro-detail enhancement
        enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)

        return enhanced
    except (ValueError, AttributeError) as e:  # type: ignore
        print(f"Error in enhance_image: {e}")
        return image


# -------------------- SUPER RESOLUTION --------------------


def super_resolution(image):  # type: ignore
    """Apply super resolution to upscale image 4x."""
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(MODEL_PATH)
        sr.setModel("espcn", 4)  # 4x upscale
        upscaled = sr.upsample(image)
        return upscaled
    except (ValueError, AttributeError, FileNotFoundError) as e:  # type: ignore
        print(f"Error in super_resolution: {e}")
        return image


# -------------------- FILTERS --------------------


def apply_filters(image):  # type: ignore
    """Apply various filters to the image."""
    try:
        results = {}

        results["grayscale"] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results["edges"] = cv2.Canny(image, 100, 200)
        results["blur"] = cv2.GaussianBlur(image, (7, 7), 0)
        results["bilateral"] = cv2.bilateralFilter(image, 9, 75, 75)

        return results
    except (ValueError, AttributeError) as e:  # type: ignore
        print(f"Error in apply_filters: {e}")
        return {}


# -------------------- MAIN PIPELINE --------------------


def main():  # type: ignore
    """Main pipeline to process all images."""
    try:
        print("Processing images...")

        if not os.path.isdir(INPUT_DIR):
            print(f"Error: Input directory '{INPUT_DIR}' not found.")
            return

        for filename in os.listdir(INPUT_DIR):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(INPUT_DIR, filename)
                image = cv2.imread(path)

                if image is None:
                    print(f"Skipped: {filename}")
                    continue

                base, ext = os.path.splitext(filename)

                # Enhancement
                enhanced = enhance_image(image)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_enhanced{ext}"), enhanced)

                # Super Resolution
                superres = super_resolution(enhanced)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_superres{ext}"), superres)

                # Filters
                filtered = apply_filters(enhanced)
                for name, img in filtered.items():
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_{name}{ext}"), img)

                print(f"Done: {filename}")

        print("\n✓ All images processed successfully.")

    except (ValueError, AttributeError, OSError) as e:  # type: ignore
        print(f"Error in main: {e}")


# -------------------- RUN --------------------

if __name__ == "__main__":
    main()

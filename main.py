"""FastAPI Image Enhancement API with OpenCV."""
# pyright: ignore[reportMissingImports, reportGeneralTypeIssues]

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2  # type: ignore
import numpy as np  # type: ignore
import uuid
from pathlib import Path

app = FastAPI()

# -------- PATH SETUP --------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = STATIC_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- TEMPLATES --------
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# -------- STATIC FILES --------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -------- IMAGE ENHANCEMENT --------
def enhance_image(image):  # type: ignore
    """Enhance image with denoise, sharpening, and CLAHE contrast boost."""
    try:
        # Mild denoise
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)  # type: ignore

        # Gentle sharpening
        blur = cv2.GaussianBlur(denoised, (0, 0), 1.0)  # type: ignore
        sharpened = cv2.addWeighted(denoised, 1.3, blur, -0.3, 0)  # type: ignore

        # Mild CLAHE contrast boost
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)  # type: ignore
        l, a, b = cv2.split(lab)  # type: ignore

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # type: ignore
        l = clahe.apply(l)  # type: ignore

        merged = cv2.merge((l, a, b))  # type: ignore
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)  # type: ignore

        return enhanced
    except (ValueError, AttributeError, TypeError) as e:  # type: ignore
        print(f"Error in enhance_image: {e}")
        return image


# -------- ROUTES --------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):  # type: ignore
    """Serve home page."""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:  # type: ignore
        print(f"Error in home route: {e}")
        return HTMLResponse("<h1>Error loading page</h1>")


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):  # type: ignore
    """Upload and enhance an image."""
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)  # type: ignore
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)  # type: ignore

        if image is None:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "Invalid image file"},
            )

        enhanced = enhance_image(image)

        filename = f"{uuid.uuid4()}.jpg"
        output_path = OUTPUT_DIR / filename

        success = cv2.imwrite(str(output_path), enhanced)  # type: ignore
        if not success:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "Failed to save enhanced image"},
            )

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "output_image": f"static/output/{filename}",
            },
        )
    except (ValueError, AttributeError, OSError) as e:  # type: ignore
        print(f"Error in upload_image: {e}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Error processing image: {str(e)}"},
        )
    except Exception as e:  # type: ignore
        print(f"Unexpected error in upload_image: {e}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "An unexpected error occurred"},
        )

import streamlit as st
import os
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageFont
import requests
from io import BytesIO
import tempfile
from rapidocr_onnxruntime import RapidOCR
import pandas as pd
import cv2
import torch
from torchvision import transforms
from torchvision.ops import nms
import re  # Import the regular expression library
import numpy as np
import altair
#from ironpdf import PdfDocument
import fitz
import io
from streamlit_elements import mui
import requests
from PIL import ImageFont

from io import BytesIO
import tempfile
from rapidocr_onnxruntime import RapidOCR
import pandas as pd
import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms
import os
import re  # Import the regular expression library
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from rapidocr_onnxruntime import RapidOCR
from torchvision import transforms
import pandas as pd
import streamlit as st
Image.MAX_IMAGE_PIXELS = None  # Removes the limit entirely

st.set_page_config(layout="wide")

base_dir = os.path.dirname(__file__)  # Dynamically get the script directory

# Relative path to your YOLOv5 directory from the base directory
yolov5_rel_path = ''

# Full path to the YOLOv5 directory
yolov5_dir = os.path.join(base_dir, yolov5_rel_path)

# Relative path to your trained model from the base directory
model_rel_path = os.path.join('runs', 'train', 'exp3', 'weights', 'best.pt')

# Full path to the trained model
model_path = os.path.join(base_dir, model_rel_path)

# Convert model_path to string - This is the necessary change
model_path_str = str(model_path)  # Convert WindowsPath or PosixPath to string

# Load the trained model with force_reload
model = torch.hub.load(yolov5_dir, 'custom', path=model_path_str, source='local', force_reload=True)





# Define your class color mapping here
CLASS_COLORS = {
    "Instrument-square": (255, 0, 0),  # Red color for Instrument-square
    "Instrument": (0, 255, 0),  # Green color for Instrument
    "Instrument-offset": (0, 0, 255),  # Blue color for Instrument-offset
    "Instrument-square-offset": (128, 0, 128),  # Yellow color for Instrument-square-offset
    # Add more classes and their colors as needed
}





def run_inference_and_get_results(confidence_threshold, img, first_nms_threshold=0.3, second_nms_threshold=0.7):
    model.conf = confidence_threshold  # Set the confidence threshold
    results = model(img)  # Run the model inference

    detected_objects = []
    boxes = []
    scores = []

    if len(results.xyxy[0]) == 0:
        print("No detections were made by the model.")
        return []

    for i in range(len(results.xyxy[0])):
        bbox = results.xyxy[0][i].cpu().numpy()
        class_id = int(bbox[5])
        class_name = model.names[class_id]
        confidence = bbox[4]

        xmin, ymin, xmax, ymax = map(int, bbox[:4].tolist())
        boxes.append([xmin, ymin, xmax, ymax])
        scores.append(confidence.item())

        detected_objects.append({
            "class": class_name,
            "confidence": confidence.item(),
            "bbox": [xmin, ymin, xmax, ymax]
        })

    if len(boxes) == 0:
        print("No boxes to process with NMS.")
        return []

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    # Debug: Print number of boxes before any NMS
    print(f"Number of boxes before any NMS: {len(boxes)}")


    # First NMS step with a larger IoU threshold to remove the most obvious overlaps
    keep_indices_first = nms(boxes_tensor, scores_tensor, first_nms_threshold)

    # Debug: Print number of boxes after the first NMS step
    print(f"Number of boxes after the first NMS step: {len(keep_indices_first)}")

    if not len(keep_indices_first):
        print("No boxes were kept after first NMS. Check IoU threshold and detections.")
        return []

    # If the second NMS step is needed, apply it. Otherwise, you can return results after the first NMS step
    if 0 < second_nms_threshold < first_nms_threshold:
        # Filter boxes and scores after the first NMS
        boxes_tensor_first_nms = boxes_tensor[keep_indices_first]
        scores_tensor_first_nms = scores_tensor[keep_indices_first]

        # Second NMS step with a smaller IoU threshold to refine the results
        keep_indices_second = nms(boxes_tensor_first_nms, scores_tensor_first_nms, second_nms_threshold)

        # Debug: Print number of boxes after the second NMS step
        print(f"Number of boxes after the second NMS step: {len(keep_indices_second)}")
        final_indices = keep_indices_first[keep_indices_second]
    else:
        # No second NMS step needed
        final_indices = keep_indices_first

    # Filter the detected_objects based on the final_indices
    detected_objects_nms = [detected_objects[i] for i in final_indices]

    # Debug: Print boxes after NMS steps
    print(f"Boxes after NMS steps: {[detected_objects[i]['bbox'] for i in final_indices]}")

    return detected_objects_nms


def draw_boxes_with_class_colors(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Load a default font

    for detection in detections:
        class_name = detection['class']
        confidence = detection['confidence']
        xmin, ymin, xmax, ymax = detection['bbox']

        # Determine color based on class_name
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

        # Draw text
        text = f"{class_name} {confidence:.2f}"
        draw.text((xmin, ymin), text, fill=color, font=font)

    return image

def crop_detected_areas(image, detections, margin=10):
    cropped_images = []
    for idx, det in enumerate(detections):
        xmin, ymin, xmax, ymax = det['bbox']
        xmin = max(0, xmin - margin)
        ymin = max(0, ymin - margin)
        xmax = min(image.width, xmax + margin)
        ymax = min(image.height, ymax + margin)
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        if cropped_image.mode == 'RGBA':
            cropped_image = cropped_image.convert('RGB')
        cropped_images.append(cropped_image)
    return cropped_images

def enhance_images(images, resize_factor, denoise_strength, denoise_template_window_size, denoise_search_window, thresholding, deskew_angle):

    enhanced_images = []
    for image in images:
        # Resize the image
        new_size = (int(image.width * resize_factor), int(image.height * resize_factor))
        resized_image = image.resize(new_size, Image.LANCZOS)

        # Convert to grayscale
        grayscale_image = ImageOps.grayscale(resized_image)

        # Apply Non-local Means Denoising
        np_grayscale = np.array(grayscale_image)
        denoised_image = cv2.fastNlMeansDenoising(np_grayscale, None, denoise_strength, denoise_template_window_size,
                                                 denoise_search_window)

        # Binarization with Otsu’s Thresholding
        if thresholding:
            _, binarized_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binarized_image = denoised_image

        # Deskewing the image
        coords = np.column_stack(np.where(binarized_image > 0))

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle
        angle += deskew_angle  # Adjust the angle based on slider input
        (h, w) = binarized_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed_image = cv2.warpAffine(binarized_image, M, (w, h), flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)

        # Sharpening the deskewed image
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(deskewed_image, -1, sharpen_kernel)

        final_image = Image.fromarray(sharpened_image)
        enhanced_images.append(final_image)
    return enhanced_images


def save_enhanced_images(images, directory=r"P-ID\yolo\enhanced_images"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    saved_image_paths = []
    for idx, image in enumerate(images):
        file_path = os.path.join(directory, f"enhanced_image_{idx}.png")
        image.save(file_path)
        saved_image_paths.append(file_path)

    return saved_image_paths

def save_cropped_images_with_classes(cropped_images, detected_objects, directory=r"P-ID\yolo\cropped_images"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    saved_image_paths = []
    for idx, (image, detected_object) in enumerate(zip(cropped_images, detected_objects)):
        class_name = detected_object['class']  # Extract class name from detection
        sanitized_class_name = re.sub('[^0-9a-zA-Z]+', '_', class_name)
        file_name = f"{sanitized_class_name}_{idx}.png"
        file_path = os.path.join(directory, file_name)
        image.save(file_path)
        saved_image_paths.append(file_path)

    return saved_image_paths

# Function to perform OCR and extract text from an image file path
def extract_text_from_image(image_path):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image_file:
            image.save(temp_image_file, format='PNG')
            temp_image_file_path = temp_image_file.name

        text = text_extractor.extract_text(temp_image_file_path)
        return text
    except Exception as e:
        return f"Error in text extraction: {e}"

# OCR extraction class
class RapidOCRTextExtractor:
    def __init__(self, engine):
        self.engine = engine

    def extract_text(self, image_path):
        img = Image.open(image_path)
        open_cv_image = pil_to_cv2(img)
        preprocessed_image = preprocess_for_ocr(open_cv_image)
        result, _ = self.engine(preprocessed_image)
        if result:
            return ' '.join([res[1] for res in result])
        return ''

# Convert PIL image to OpenCV format
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Preprocess image for OCR
def preprocess_for_ocr(image, target_size=(300, 300)):
    # Resize image
    image = cv2.resize(image, target_size)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better contrast
    preprocessed_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return preprocessed_image

# Initialize OCR engine for Streamlit app
ocr_engine = RapidOCR()
text_extractor = RapidOCRTextExtractor(ocr_engine)


# Initialize session state
if 'detected_objects' not in st.session_state:
    st.session_state['detected_objects'] = []
if 'valid_instruments' not in st.session_state:
    st.session_state['valid_instruments'] = pd.DataFrame(columns=["Instrument Type", "Valid Instrument Tags", "Image", "is_valid"])
if 'cropped_images' not in st.session_state:
    st.session_state['cropped_images'] = []
if 'enhanced_images' not in st.session_state:
    st.session_state['enhanced_images'] = []


def generate_regex_pattern_from_parts(system, function, sequence):
    """
    Generates a regex pattern from the system number, function, and sequence number.
    The pattern will allow the system number to be at the beginning or at the end of the instrument number.
    """
    system_regex = rf"({system})?" if system else ""
    function_regex = rf"{function}" if function else ""
    sequence_regex = rf"(\d{{4}}[A-Z]{{0,2}})" if sequence else ""

    # Combine parts into a comprehensive regex pattern allowing system number at the start or end
    full_regex = f"({system_regex}-{function_regex}-{sequence_regex}|{function_regex}-{sequence_regex}-{system_regex})"
    return rf"\b{full_regex}\b"


def format_instrument_number(number):
    # Split the number using a regex that finds the system number, function code, and sequence
    parts = re.match(r"(\d+)-([A-Z]+)-(\d+[A-Z]*)", number)
    if parts:
        formatted_number = f"{parts.group(1)}-{parts.group(2)}-{parts.group(3)}"
        return formatted_number
    return number  # Return original number if it doesn't match the expected pattern



def render_pdf_page_to_png(file, dpi=400):
    """
    Render each page of a PDF as a whole image in PNG format.

    Args:
    - file: The uploaded PDF file.
    - dpi: The dots per inch (resolution) of the rendered image.

    Returns:
    A list of PIL images in PNG format, each representing a page of the PDF.
    """
    images = []
    doc = fitz.open(stream=file.read())  # Open the PDF file
    for page_num in range(len(doc)):  # Iterate through each page
        page = doc.load_page(page_num)  # Load the current page
        pix = page.get_pixmap(dpi=dpi)  # Render the page to a pixmap at the specified DPI

        img_bytes = io.BytesIO(pix.tobytes("png"))  # Convert pixmap to PNG bytes and then to BytesIO

        image = Image.open(img_bytes)  # Read the image from the BytesIO buffer
        images.append(image)
    return images




def render_pdf_page_to_jpg_with_poppler(file, dpi=1200):
    """
    Render each page of a PDF as a whole image in JPG format using Poppler through pdf2image.

    Args:
    - file: The uploaded PDF file, as a file-like object.
    - dpi: The dots per inch (resolution) of the rendered image.

    Returns:
    A list of PIL images in JPG format, each representing a page of the PDF.
    """
    images = []

    # Convert PDF file to images using pdf2image
    pil_images = convert_from_bytes(file.read(), dpi=dpi)

    # Convert each PIL image to JPG and append to the images list
    for img in pil_images:
        jpg_bytes = io.BytesIO()
        img.save(jpg_bytes, format="JPEG")
        jpg_bytes.seek(0)
        jpg_image = Image.open(jpg_bytes)
        images.append(jpg_image)

    return images



def render_pdf_page_to_jpg(file, dpi=1200):
    """
    Render each page of a PDF as a whole image in JPG format.

    Args:
    - file: The uploaded PDF file.
    - dpi: The dots per inch (resolution) of the rendered image.

    Returns:
    A list of PIL images in JPG format, each representing a page of the PDF.
    """
    images = []
    doc = fitz.open(stream=file.read())  # Open the PDF file
    for page_num in range(len(doc)):  # Iterate through each page
        page = doc.load_page(page_num)  # Load the current page
        pix = page.get_pixmap(dpi=dpi)  # Render the page to a pixmap at the specified DPI

        img_bytes = io.BytesIO(pix.tobytes("png"))  # Convert pixmap to PNG bytes and then to BytesIO
        img = Image.open(img_bytes)  # Create a PIL Image from the PNG data
        jpg_bytes = io.BytesIO()  # Create a new BytesIO object for the JPG data
        img.save(jpg_bytes, format="JPEG")  # Save the PIL Image as JPEG to the new BytesIO object
        jpg_bytes.seek(0)  # Rewind the BytesIO object for reading

        jpg_image = Image.open(jpg_bytes)  # Create a new PIL Image from the JPG data
        images.append(jpg_image)
    return images


def render_pdf_page_to_png_with_mupdf(file_stream, dpi=800):
    """
    Render each page of a PDF as a whole image in PNG format using MuPDF.

    Args:
    - file_stream: The byte stream of the uploaded PDF file.
    - dpi: The resolution of the rendered image in dots per inch (DPI).

    Returns:
    A list of PIL Image objects, each representing a page of the PDF rendered as a PNG image.
    """
    images = []

    # Ensure the stream position is at the beginning
    file_stream.seek(0)

    # Open the PDF file from the stream
    doc = fitz.open("pdf", file_stream.read())

    for page_num in range(len(doc)):  # Iterate through each page
        page = doc.load_page(page_num)  # Load the current page
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # A matrix for scaling images according to the specified DPI
        pix = page.get_pixmap(matrix=mat, alpha=False)  # Render the page to a pixmap

        # Directly convert pixmap to PNG bytes
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))  # Create a PIL Image from the PNG bytes

        # Ensure the image is in RGBA format to match the bit depth of training images
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        images.append(img)
    return images





def draw_boxes_with_class_colors(image, detections, line_width=3):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Use a default font

    for detection in detections:
        class_name = detection['class']
        confidence = detection['confidence']
        xmin, ymin, xmax, ymax = detection['bbox']

        # Determine color based on class_name
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        # Draw rectangle with increased line width
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)

        # Draw text
        text = f"{class_name} {confidence:.2f}"
        draw.text((xmin, ymin), text, fill=color, font=font)

    return image






def render_pdf_to_images(uploaded_file):
    """
    Render each page of a PDF to images using PyMuPDF (fitz).

    Args:
    - uploaded_file: The uploaded PDF file as a BytesIO object.

    Returns:
    - List of PIL Image objects, one per page.
    """
    images = []
    # Open the PDF file from the uploaded BytesIO object
    doc = fitz.open(stream=uploaded_file.read())
    for page in doc:
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    return images



page = st.sidebar.selectbox("Select a page", ("Object Detection", "OCR"))


# Assuming render_pdf_to_images is defined elsewhere in your code

if page == "Object Detection":
    st.write("""
    ## Object Detection 🔍
    Use the Object Detection feature to automatically identify and label different instruments and components in your P&ID diagrams. Adjust detection settings as needed.
    - **Detect Object:** Start object detection on your uploaded image or PDF.
    - **Enhancement:** Enhance detected objects for better analysis.
    """)

    # Define a tab option for different functionalities
    tab_option = st.radio("Select Option", ["Detect Object", "Image Enhancement Tool"], horizontal=True)

    if tab_option == "Detect Object":
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "png", "jpeg", "pdf"])

        if uploaded_file is not None:
            # Store the uploaded file in session state for access in other tabs
            st.session_state['uploaded_file'] = uploaded_file

            images = []  # Initialize an empty list for storing images from either an uploaded image or PDF

            if uploaded_file.type == "application/pdf":
                with st.spinner("Processing PDF..."):
                    images = render_pdf_page_to_png_with_mupdf(uploaded_file)
                    for image in images:
                        st.image(image, caption="Image from PDF", use_column_width=True)
            else:
                image = Image.open(uploaded_file).convert('RGB')
                images.append(image)  # Add the single image to the list
                st.image(image, caption='Uploaded Image', use_column_width=True)

            # Store images in session state for access in other tabs
            st.session_state['images'] = images

            if images:  # This will check if the images list is not empty
                if st.button('Detect Objects'):
                    for image in images:
                        # Run model inference to get detections for each image
                        detections = run_inference_and_get_results(confidence_threshold, image)
                        if detections:
                            image_with_boxes = draw_boxes_with_class_colors(image.copy(), detections)
                            st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)

                            # Crop detected areas from the image
                            cropped_images = crop_detected_areas(image, detections)
                            st.session_state['detected_objects'] = detections
                            st.session_state['cropped_images'] = cropped_images

                            # Display cropped images
                            st.success("Cropped objects:")
                            cols = st.columns(len(cropped_images))
                            for idx, col in enumerate(cols):
                                with col:
                                    st.image(cropped_images[idx], caption=f'Cropped Object {idx + 1}', width=100)
                        else:
                            st.info("No objects detected.")

    if tab_option == "Image Enhancement Tool":
        # Main expander for "Improve Detected Objects Quality"
        with st.expander("Improve Detected Objects Quality"):
            if 'cropped_images' in st.session_state and len(st.session_state['cropped_images']) > 0:
                st.success("Detected objects:")
                cols = st.columns(len(st.session_state['cropped_images']))
                for idx, col in enumerate(cols):
                    with col:
                        st.image(st.session_state['cropped_images'][idx], caption=f'Detected Object {idx + 1}', width=100)


                col1, col2 = st.columns(2)
                with col1:
                    resize_factor = st.slider("Resize Factor", min_value=0.1, max_value=9.0, value=1.0, step=0.1, key='resize_factor1')
                    denoise_strength = st.slider("Denoise Strength", min_value=0, max_value=200, value=10, key='denoise_strength1')
                    denoise_template_window_size = st.slider("Denoise Template Window Size", min_value=3, max_value=41, value=7, step=2, key='denoise_template_window_size1')
                    denoise_search_window = st.slider("Denoise Search Window", min_value=3, max_value=41, value=21, step=2, key='denoise_search_window1')
                    thresholding = st.checkbox("Thresholding", value=True, key='thresholding1')
                    deskew_angle = st.slider("Deskew Angle", min_value=-90, max_value=90, value=0, key='deskew_angle1')

                    if st.button('Apply Enhancements', key='apply_enhancements1'):
                        # Assume enhance_images function definition is provided elsewhere
                        enhanced_images = enhance_images(
                            st.session_state['cropped_images'],
                            resize_factor=resize_factor,
                            denoise_strength=denoise_strength,
                            denoise_template_window_size=denoise_template_window_size,
                            denoise_search_window=denoise_search_window,
                            thresholding=thresholding,
                            deskew_angle=deskew_angle
                        )
                        st.session_state['enhanced_images'] = enhanced_images
                        st.success("Enhancement parameters applied successfully.")
                with col2:
                    if 'enhanced_images' in st.session_state and len(st.session_state['enhanced_images']) > 0:
                        st.image(st.session_state['enhanced_images'][0], caption='Enhanced Object 1')
            else:
                st.warning("No cropped images to display. Please detect objects first.")



        if 'images' in st.session_state and len(st.session_state['images']) > 0:
            with st.expander("Improve P&ID Image Quality"):
                # Assuming st.session_state['images'] stores PIL images or image file paths
                # Display the first uploaded image from the session state directly
                st.image(st.session_state['images'][0], caption='Uploaded Image', use_column_width=True)

                # Assuming the first image from the session state is the one we want to enhance
                image = st.session_state['images'][0]  # Directly use the stored image

                img_center_x, img_center_y = image.width // 2, image.height // 2

                if 'x_coord' not in st.session_state or 'y_coord' not in st.session_state:
                    st.session_state['x_coord'] = img_center_x
                    st.session_state['y_coord'] = img_center_y

                col1, col2 = st.columns([7, 3])
                with col1:
                    # No need to display the image again as it's already displayed above
                    pass  # Placeholder in case you want to add other content here

                with col2:
                    new_x_coord = st.number_input("X coordinate of interest", min_value=0, max_value=image.width, value=st.session_state['x_coord'], key='x_coord2')
                    new_y_coord = st.number_input("Y coordinate of interest", min_value=0, max_value=image.height, value=st.session_state['y_coord'], key='y_coord2')
                    new_region_size = st.number_input("Size of the region to zoom in on", min_value=50, max_value=min(image.width, image.height), value=100, key='new_region_size2')

                    resize_factor = st.slider("Resize Factor", min_value=0.1, max_value=9.0, value=1.0, step=0.1, key='resize_factor2')
                    denoise_strength = st.slider("Denoise Strength", min_value=0, max_value=200, value=10, key='denoise_strength2')
                    denoise_template_window_size = st.slider("Denoise Template Window Size", min_value=3, max_value=41, value=7, step=2, key='denoise_template_window_size2')
                    denoise_search_window = st.slider("Denoise Search Window", min_value=3, max_value=41, value=21, step=2, key='denoise_search_window2')
                    thresholding = st.checkbox("Thresholding", value=True, key='thresholding2')
                    deskew_angle = st.slider("Deskew Angle", min_value=-90, max_value=90, value=0, key='deskew_angle2')

                    st.session_state['x_coord'] = new_x_coord
                    st.session_state['y_coord'] = new_y_coord

                    left = max(0, new_x_coord - new_region_size // 2)
                    top = max(0, new_y_coord - new_region_size // 2)
                    right = left + new_region_size
                    bottom = top + new_region_size
                    region_of_interest = image.crop((left, top, right, bottom))

                    # Assume enhance_images function definition is provided elsewhere
                    enhanced_region = enhance_images(
                        [region_of_interest],
                        resize_factor,
                        denoise_strength,
                        denoise_template_window_size,
                        denoise_search_window,
                        thresholding,
                        deskew_angle
                    )[0]

                with col1:
                    st.image(enhanced_region, caption='Zoomed and Enhanced Region', use_column_width=True)
        else:
            st.warning("No image uploaded. Please go to the 'Detect Object' tab and upload an image first.")


if page == "OCR":
    st.sidebar.write("""
    ## OCR (Optical Character Recognition) 📖
    Extract text from detected objects in P&ID diagrams. Customize your extraction with regular expressions for precise analysis.
    - **Regex Generator:** Create custom regex patterns for extracting specific instrument numbers.
    - **Extract Text:** Use OCR to extract text from images of detected objects. Apply custom regex for filtering.
    """)
    st.subheader("Optical Character Recognition using RapidOCR")

    selection = st.radio("Select Option", ["Regex Generator", "Extract Text"], horizontal=True)

    if selection == "Regex Generator":
        st.subheader("Regex Pattern Generator")
        company_selection = st.selectbox(
            "For naming convention: 00-AA-1234; AA-00-1234",
            ["Naming convention"]
        )

        if company_selection == "Naming convention": # correct regex \b(\d{2})?[,.)]?\s*([A-Z]{2,5})\s*(\d{4})(?:\s*[A-Z])?\b
            with st.form("regex_generator_eigen"):
                st.write("Details for naming convention:")
                num_system_digits_eigen = st.number_input("System digits (default 3):", min_value=1, max_value=5, value=3, step=1, key="eigen_system")
                num_function_letters_eigen = st.number_input("Function Code letters (2-5):", min_value=2, max_value=5, value=2, step=1, key="eigen_function")
                num_loop_sequence_digits_eigen = st.number_input("Loop Sequence digits (default 4):", min_value=1, max_value=6, value=4, step=1, key="eigen_sequence")
                submitted_eigen = st.form_submit_button("Generate Pattern")

                if submitted_eigen:
                    eigen_pattern = fr'\b(\d{{{num_system_digits_eigen}}})?[,.)]?\s*([A-Z]{{{num_function_letters_eigen},5}}\s*\d{{{num_loop_sequence_digits_eigen}}}(?:\s*[A-Z])?)\b'
                    st.write("Generated regex pattern:")
                    st.code(eigen_pattern)



    if selection == "Extract Text":
        st.subheader("Extract Text from Cropped Images")

        # Checkbox to choose between seeing all text or filtered by regex
        use_regex = st.checkbox("Filter text using a custom regex pattern", True)

        if use_regex:
            default_pattern = r'\b(\d{2})?[,.)]?\s*([A-Z]{2,5})\s*(\d{4})(?:\s*[A-Z])?\b'
            regex_pattern = st.text_input("Enter the custom regex pattern for instrument numbers:", value=default_pattern)
        else:
            regex_pattern = r".*"  # Match anything if regex is not used

        system_number_hint = st.text_input("System Number Hint:", key="system_number_hint")
        system_number_position = st.radio(
            "Select where to place the System Number:",
            ('Beginning', 'End'),
            index=0,
            key="system_number_position"
        )

        if st.button('Extract Instruments'):
            extracted_data = []
            all_text_data = []  # List to hold all extracted text for display
            if 'detected_objects' in st.session_state and 'cropped_images' in st.session_state:
                for idx, detected_object in enumerate(st.session_state['detected_objects']):
                    class_name = detected_object['class']  # Extract the class name
                    image = st.session_state['cropped_images'][idx]  # Accessing the cropped image directly

                    # Perform OCR on each cropped image
                    text = extract_text_from_image(image)
                    all_text_data.append(text)  # Add all extracted text to the list
                    if use_regex:
                        instrument_pattern = re.compile(regex_pattern)
                        matches = instrument_pattern.findall(text)
                    else:
                        matches = [text]  # Use the entire extracted text if not using regex

                    for match in matches:
                        # Your existing processing logic here...
                        pass

                st.session_state['extracted_data'] = extracted_data

                if not use_regex:
                    st.subheader("Extracted Text from Images")
                    for i, text in enumerate(all_text_data, start=1):
                        st.text_area(f"Text from Image {i}", value=text, height=100)
            else:
                st.warning("No detected objects or cropped images found in the session state.")


        # Displaying and editing the extracted instrument numbers
        if 'extracted_data' in st.session_state and st.session_state['extracted_data']:
            for data in st.session_state['extracted_data']:
                edited_tagname = st.text_input(
                    f"Edit Tagname {data['Index']}",
                    value=data.get('TAGNAME', 'N/A'),  # Default value 'N/A' if 'TAGNAME' not found
                    key=f"edit_tag_{data['Index']}"
                )
                # Update the TAGNAME in extracted_data with edited_tagname if needed
                data['TAGNAME'] = edited_tagname

            if st.button("Confirm Edits"):
                # Process the confirmed edits here if necessary
                st.success("Instrument numbers and Tagnames updated.")

        # Displaying the table with extracted data
        if st.button("Show Table"):
            if 'extracted_data' in st.session_state and st.session_state['extracted_data']:
                df = pd.DataFrame(st.session_state['extracted_data'])
                column_order = ["Index", "Original Number", "TAGNAME", "CLASS", "SYSTEM", "FUNCTION_CODE", "LOOP SEQUENCE", "DRAWING_NO"]
                df = df[column_order]  # Reorder columns
                st.dataframe(df)
            else:
                st.warning("No data available. Please extract and edit instrument numbers and Tagnames first.")
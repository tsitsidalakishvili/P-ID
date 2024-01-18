import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
from rapidocr_onnxruntime import RapidOCR
import tempfile
import pandas as pd
import re  # Import the regular expression library
import torch

st.set_page_config(layout="wide")

# Load your pre-trained model
import torch

# Load your pre-trained model
model_path = r'C:\Users\Tsitsi\Desktop\experiments\Demo\yolov5\runs\train\exp3\weights\best.pt'
hubconf_path = r'C:\Users\Tsitsi\Desktop\experiments\Demo\yolov5'
model = torch.hub.load(hubconf_path, 'custom', path=model_path, source='local')

# Function to run inference and return results
def run_inference_and_get_results(confidence_threshold, img):
    model.conf = confidence_threshold  # Set the confidence threshold
    results = model(img)  # Run the model inference
    detected_objects = []
    for i in range(len(results.xyxy[0])):
        bbox = results.xyxy[0][i].cpu().numpy()
        class_id = int(bbox[5])
        class_name = model.names[class_id]
        confidence = bbox[4]
        xmin, ymin, xmax, ymax = map(int, bbox[:4].tolist())
        detected_objects.append({
            "class": class_name,
            "confidence": confidence.item(),
            "bbox": [xmin, ymin, xmax, ymax]
        })
    return detected_objects

# Function to crop detected areas
def crop_detected_areas(image, detections, margin=2):
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

# Function to enhance images with dynamic parameters
def enhance_images(images, resize_factor, denoise_strength, denoise_template_window_size, denoise_search_window,
                   thresholding, deskew_angle):
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

# Function to save enhanced images to a specified directory
def save_enhanced_images(images, directory=r"C:\Users\Tsitsi\Desktop\experiments\Demo\yolo\enhanced_images"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    saved_image_paths = []
    for idx, image in enumerate(images):
        file_path = os.path.join(directory, f"enhanced_image_{idx}.png")
        image.save(file_path)
        saved_image_paths.append(file_path)

    return saved_image_paths

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

# Function to perform OCR and extract text from an image file path
def extract_text_from_image(image):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image_file:
            image.save(temp_image_file, format='PNG')
            temp_image_file_path = temp_image_file.name

        text = text_extractor.extract_text(temp_image_file_path)
        return text
    except Exception as e:
        return f"Error in text extraction: {e}"



# Streamlit interface
st.title("P&ID Object Detection and OCR Analysis")

# Sidebar for navigation
page = st.sidebar.selectbox("Select a page", ("Object Detection", "OCR", "Combine Results"))

uploaded_file = None  # Add this line to declare the variable

# Object Detection page
if page == "Object Detection":
    st.subheader("Object Detection using YOLOv5")

    # Confidence Threshold Slider
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    # Initialize session state for cropped images
    if 'cropped_images' not in st.session_state:
        st.session_state['cropped_images'] = []

    # Create an expander for uploaded file section
    uploaded_file_expander = st.expander("Uploaded Image", expanded=True)
    with uploaded_file_expander:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

    # Detect Objects Button
    if st.button('Detect Objects'):
        uploaded_file_expander.expanded = False  # Collapse the uploaded file section
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')

            # Object Detection
            detections = run_inference_and_get_results(confidence_threshold, image)

            # Store the detected objects in session state
            st.session_state['detected_objects'] = detections

            if detections:
                # Create a copy of the image to draw on
                image_with_boxes = image.copy()
                draw = ImageDraw.Draw(image_with_boxes)
                for det in detections:
                    draw.rectangle(det['bbox'], outline="red", width=2)
                    draw.text(det['bbox'][:2], f"{det['class']} ({det['confidence']:.2f})", fill="white")

                # Display the image with detected objects
                st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)

                # Crop detected areas and store in session state
                st.session_state['cropped_images'] = crop_detected_areas(image, detections)
                st.success("Displaying cropped detected objects:")

                # Create a multi-column layout to display cropped images
                cols = st.columns(len(st.session_state['cropped_images']))  # Create as many columns as cropped images
                for idx, col in enumerate(cols):
                    with col:
                        st.image(st.session_state['cropped_images'][idx], caption=f'Detected Object {idx + 1}',
                                 width=100)  # Adjust width as needed
            else:
                st.error("No objects detected. Please upload a different image.")


if page == "OCR":
    st.subheader("Optical Caracter Recognition using RapidOCR")

    # Create a layout with two columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Enhancement Parameters")
        # Enhancement Parameter Sliders (After Detecting and Enhancing Cropped Images)
        resize_factor = st.slider("Resize Factor", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        denoise_strength = st.slider("Denoise Strength", min_value=0, max_value=200, value=10)
        denoise_template_window_size = st.slider("Denoise Template Window Size", min_value=3, max_value=41, value=7,
                                                 step=2)
        denoise_search_window = st.slider("Denoise Search Window", min_value=3, max_value=41, value=21, step=2)
        thresholding = st.checkbox("Thresholding", value=True)
        deskew_angle = st.slider("Deskew Angle", min_value=-90, max_value=90, value=0)

        # Enhance Cropped Images Button (Using Default Parameters)
        if st.button('Apply'):
            if st.session_state['cropped_images']:
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
                st.success("Enhancement parameters applied successfully!")

    with col2:
        if 'enhanced_images' in st.session_state and st.session_state['enhanced_images']:
            # Display only the first enhanced image
            st.image(st.session_state['enhanced_images'][0], caption='Enhanced Object 1', use_column_width=True)

    # Regular expression pattern for instrument numbers based on the tag format NN AA(AAA) NNNN (A)
    # This assumes that there are no spaces within each part of the tag
    #instrument_pattern = re.compile(r'\b(\d{2})\s+([A-Z]{2,5})\s+(\d{4})(?:\s+([A-Z]))?\b')

    instrument_pattern = re.compile(r'\b(\d{2}\s*[A-Z]{2,5}\s*\d{4}(?:\s*[A-Z])?)\b')


    # Initialize data outside the button click block
    data = []

    # Streamlit interface - Display Enhanced Images with Text Button
    if st.button('Extract Instruments'):
        if 'enhanced_images' in st.session_state and st.session_state['enhanced_images']:
            enhanced_images = st.session_state['enhanced_images']
            detected_objects = st.session_state.get('detected_objects', [])

            for idx, image in enumerate(enhanced_images):
                text = extract_text_from_image(image)

                # Apply regular expression to find and validate instrument numbers
                matches = instrument_pattern.findall(text)
                # Filter and format matches to ensure they follow the exact tag pattern
                valid_instrument_numbers = [''.join(match).strip() for match in matches if match]

                # Joining valid instrument numbers with a comma
                instrument_numbers = ', '.join(valid_instrument_numbers)

                # Check if the index is within the bounds of detected_objects
                if idx < len(detected_objects):
                    class_name = detected_objects[idx]['class']
                else:
                    class_name = f'Enhanced Object {idx + 1}'

                # Append only valid instrument numbers to the data list
                if valid_instrument_numbers:
                    data.append((class_name, instrument_numbers, image))
                else:
                    data.append((class_name, 'No valid tag found', image))

            # Save valid instrument data in session state
            st.session_state['valid_instruments'] = pd.DataFrame(data, columns=["Instrument Type", "Valid Instrument Tags", "Image"])



            # Create a DataFrame
            df = pd.DataFrame(data, columns=["Instrument Type", "Valid Instrument Tags", "Image"])

            # Separate valid instruments and invalid images
            valid_instruments = df[df['Valid Instrument Tags'] != 'No valid tag found']
            invalid_images = df[df['Valid Instrument Tags'] == 'No valid tag found']

            # Display valid instrument numbers in one table
            if not valid_instruments.empty:
                st.subheader("Valid Instrument Numbers")
                st.table(valid_instruments[["Instrument Type", "Valid Instrument Tags"]])

                # Display images for valid instruments vertically
                st.subheader("Images for Valid Instruments")
                valid_image_urls = valid_instruments["Image"].tolist()
                captions = valid_instruments["Instrument Type"].tolist()
                col1, col2 = st.columns(2)
                for i in range(0, len(valid_image_urls), 2):
                    with col1:
                        if i < len(valid_image_urls):
                            st.image(valid_image_urls[i], caption=captions[i], width=150)
                    with col2:
                        if i + 1 < len(valid_image_urls):
                            st.image(valid_image_urls[i + 1], caption=captions[i + 1], width=150)

            # Display non-valid images and "No valid tag found"
            if not invalid_images.empty:
                st.subheader("Invalid Images and No Valid Tags Found")
                st.table(invalid_images[["Instrument Type", "Valid Instrument Tags"]])

                # Display images for invalid images vertically
                st.subheader("Images for Invalid Images")
                invalid_image_urls = invalid_images["Image"].tolist()
                invalid_captions = invalid_images["Instrument Type"].tolist()
                for i in range(len(invalid_image_urls)):
                    st.image(invalid_image_urls[i], caption=invalid_captions[i], width=150)

        else:
            st.error("No enhanced images available. Please enhance the images first.")


# New section for "Combine Results" page
if page == "Combine Results":
    st.subheader("Combine Results")

    # Display the Valid Instrument Numbers table
    if 'valid_instruments' in st.session_state and st.session_state['valid_instruments'] is not None and not st.session_state['valid_instruments'].empty:
        st.subheader("Valid Instrument Numbers")
        valid_instruments_df = st.session_state['valid_instruments'][["Instrument Type", "Valid Instrument Tags"]]
        st.table(valid_instruments_df)

        # Get the original image
        original_image = st.session_state.get('uploaded_image', None)

        # Check if original image exists and detected objects are available
        if original_image is not None and 'detected_objects' in st.session_state:
            detections = st.session_state['detected_objects']

            # Create a copy of the original image to draw on
            image_with_boxes = original_image.copy()
            draw = ImageDraw.Draw(image_with_boxes)
            for det in detections:
                draw.rectangle(det['bbox'], outline="red", width=2)
                draw.text(det['bbox'][:2], f"{det['class']} ({det['confidence']:.2f})", fill="white")

            # Display the image with detected objects
            st.subheader("Image with Detected Objects")
            st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)

        else:
            st.warning("No detected objects available. Please ensure you have run object detection.")

    else:
        st.warning("No valid instrument data available. Please ensure you have valid instrument data.")

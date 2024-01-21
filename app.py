import os
import re  # Import the regular expression library
import tempfile

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from rapidocr_onnxruntime import RapidOCR
from torchvision import transforms
from torchvision.ops import nms

import streamlit as st



st.set_page_config(layout="wide")


# Load your pre-trained model
model_path = r'yolov5\runs\train\exp3\weights\best.pt'
hubconf_path = r'yolov5'
model = torch.hub.load(hubconf_path, 'custom', path=model_path, source='local')

# Function to run inference and return results
# Function to run inference and return results
def run_inference_and_get_results(confidence_threshold, img, nms_threshold=0.1):
    model.conf = confidence_threshold  # Set the confidence threshold
    results = model(img)  # Run the model inference

    detected_objects = []
    boxes = []
    scores = []
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



    # Ensure boxes is a 2D tensor with shape [num_boxes, 4]
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    # Debug: Print boxes before NMS
    print(f"Boxes before NMS: {boxes_tensor.size()}")

    # Apply Non-Maximum Suppression
    keep_indices = nms(boxes_tensor, scores_tensor, nms_threshold)
    

    
    # Filter the detected_objects based on the keep_indices from NMS
    detected_objects_nms = [detected_objects[i] for i in keep_indices]

    # Debug: Print boxes after NMS
    print(f"Boxes after NMS: {[detected_objects[i]['bbox'] for i in keep_indices]}")

    return detected_objects_nms


# Function to crop detected areas
def crop_detected_areas(image, detections, margin=8):
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
def save_enhanced_images(images, directory=r"C:\Users\Tsitsi\Desktop\experiments\P-ID\yolo\enhanced_images"):
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


# Initialize session state
if 'detected_objects' not in st.session_state:
    st.session_state['detected_objects'] = []
if 'valid_instruments' not in st.session_state:
    st.session_state['valid_instruments'] = pd.DataFrame(columns=["Instrument Type", "Valid Instrument Tags", "Image", "is_valid"])
if 'cropped_images' not in st.session_state:
    st.session_state['cropped_images'] = []
if 'enhanced_images' not in st.session_state:
    st.session_state['enhanced_images'] = []

st.title("P&ID Object Detection and OCR Analysis")

# Sidebar for navigation
page = st.sidebar.selectbox("Select a page", ("Object Detection", "OCR"))

# Object Detection page
if page == "Object Detection":
    st.subheader("Object Detection using YOLOv5")

    # Confidence Threshold Slider
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

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

            # Update the detected objects session state
            st.session_state.detected_objects = detections

            if detections:
                # Display the number of detected objects
                st.success(f"Detected Objects: {len(detections)}")

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
            st.success(f"Detected Objects: {len(detections)}")
        
    else:
        st.error("No objects detected. Please upload a different image.")




# OCR Page
if page == "OCR":
    st.subheader("Optical Character Recognition using RapidOCR")
    
    # Check if there are cropped images to display
    if 'cropped_images' in st.session_state and st.session_state['cropped_images']:
        st.success("Displaying cropped detected objects:")
        # Create a multi-column layout to display cropped images
        cols = st.columns(len(st.session_state['cropped_images']))  # Create as many columns as cropped images
        for idx, col in enumerate(cols):
            with col:
                st.image(st.session_state['cropped_images'][idx], caption=f'Detected Object {idx + 1}', width=100)  # Adjust width as needed
    else:
        st.warning("No cropped images to display. Please go to the 'Object Detection' page and detect objects first.")
        




        
    # Enhancement Parameters section
    enhancement_expander = st.expander("Enhancement Parameters", expanded=True)
    with enhancement_expander:


        # Create a layout with two columns for enhancement parameters and the enhanced image
        col1, col2 = st.columns(2)

        with col1:
            # Enhancement Parameter Sliders
            resize_factor = st.slider("Resize Factor", min_value=0.1, max_value=9.0, value=1.0, step=0.1)
            denoise_strength = st.slider("Denoise Strength", min_value=0, max_value=200, value=10)
            denoise_template_window_size = st.slider("Denoise Template Window Size", min_value=3, max_value=41, value=7, step=2)
            denoise_search_window = st.slider("Denoise Search Window", min_value=3, max_value=41, value=21, step=2)
            thresholding = st.checkbox("Thresholding", value=True)
            deskew_angle = st.slider("Deskew Angle", min_value=-90, max_value=90, value=0)

            # Enhance Cropped Images Button
            if st.button('Apply Enhancements'):
                if 'cropped_images' in st.session_state and st.session_state['cropped_images']:
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
            # Display the first enhanced image
            if 'enhanced_images' in st.session_state and st.session_state['enhanced_images']:
                st.image(st.session_state['enhanced_images'][0], caption='Enhanced Object 1', use_column_width=True)






    extract_text_expander = st.expander("Extract Instrument Numbers", expanded=False)
    with extract_text_expander:
        # Adjusted regex based on the naming conventions from the uploaded image
        # The pattern NN AA(AAA) NNNN (A) translates to 2 digits, followed by 2-5 uppercase letters, followed by 4 digits, and optionally followed by a single uppercase letter in parentheses
        instrument_pattern = re.compile(r'\b(\d{2}\s+[A-Z]{2,5}\s+\d{4}(?:\s+\([A-Z]\))?)\b')

        # Checkbox to toggle regex filtering based on naming conventions
        use_naming_convention = st.checkbox("Use ReGex - NN AA(AAA) NNNN (A) ", key='use_naming_convention')

        if st.button('Extract Instruments'):

            # Use cropped images or enhanced images based on whether enhancements have been applied
            images_to_use = st.session_state['enhanced_images'] if 'enhanced_images' in st.session_state and st.session_state['enhanced_images'] else st.session_state['cropped_images']
            detected_objects = st.session_state.get('detected_objects', [])

            if images_to_use:
                data = []
                for idx, image in enumerate(images_to_use):
                    text = extract_text_from_image(image)

                    if idx < len(detected_objects):
                        class_name = detected_objects[idx]['class']
                    else:
                        class_name = f'Object {idx + 1}'

                    if use_naming_convention:
                        # Find all matches and make them bold
                        matches = instrument_pattern.findall(text)
                        for match in matches:
                            text = re.sub(match, f'**{match}**', text)

                    data.append((class_name, text, image))

                # Create a DataFrame
                df = pd.DataFrame(data, columns=["Instrument Type", "Extracted Text", "Image"])

                # Display extracted text with bold valid instrument numbers
                st.subheader("Extracted Instrument Numbers")
                for _, row in df.iterrows():
                    # Using markdown with unsafe_allow_html to render bold
                    st.markdown(f"**{row['Instrument Type']}:** {row['Extracted Text']}", unsafe_allow_html=True)
                    st.image(row['Image'], width=150)
            else:
                st.error("No images available for text extraction. Please upload and detect objects first.")

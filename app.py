import streamlit as st
import os
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageFont
import torch
from torchvision.ops import nms
import re
import numpy as np
import cv2
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import threading
import datetime
from streamlit_tensorboard import st_tensorboard
from rapidocr_onnxruntime import RapidOCR

import os
import torch
import streamlit as st
import datetime

# Function to detect the environment
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Function to list available models in the directory
def list_available_models(model_dir):
    return [f for f in os.listdir(model_dir) if f.endswith('.pt')]

# Function to load YOLOv5 model
@st.cache_resource
def load_model(model_path, yolov5_dir):
    model = torch.hub.load(yolov5_dir, 'custom', path=model_path, source='local')
    return model

# Initialize Streamlit app and set page configuration
st.set_page_config(layout="wide")

if is_colab():
    # Colab specific setup
    from google.colab import drive
    drive.mount('/content/drive')
    #%cd /content/drive/MyDrive/P-ID/yolov5
    
    # Define the model directory and YOLOv5 directory in Colab
    model_dir = '/content/drive/MyDrive/P-ID/yolov5/runs'
    yolov5_dir = '.'
else:
    # Local (VS Code) specific setup
    base_dir = os.path.dirname(__file__)
    yolov5_dir = os.path.join(base_dir, 'yolov5')  # Ensure this path is correct
    model_dir = os.path.join(yolov5_dir, 'runs', 'train', 'exp2', 'weights')

# List available models in the directory
available_models = list_available_models(model_dir)

# Create a dropdown in Streamlit for model selection
selected_model = st.selectbox("Select a Model", available_models)

if selected_model:
    # Define the path for the selected model
    model_path = os.path.join(model_dir, selected_model)

    # Load the YOLOv5 model based on user selection
    model = load_model(model_path, yolov5_dir)

    st.success(f"Model '{selected_model}' loaded successfully!")

else:
    st.warning("No model selected. Please select a model to proceed.")







# Function to extract metrics from TensorBoard logs
@st.cache_data
def extract_metrics(logdir):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    metrics = {}
    try:
        metrics['accuracy'] = event_acc.Scalars('accuracy')[-1].value
        metrics['precision'] = event_acc.Scalars('precision')[-1].value
        metrics['recall'] = event_acc.Scalars('recall')[-1].value
    except KeyError:
        st.error("Could not find some metrics in the TensorBoard logs.")
    return metrics

# Function to display key metrics
def display_metrics(metrics):
    st.write("## Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%" if 'accuracy' in metrics else "N/A")
    col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%" if 'precision' in metrics else "N/A")
    col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%" if 'recall' in metrics else "N/A")

# Define class color mapping
CLASS_COLORS = {
    "Instrument-square": (255, 0, 0),
    "Instrument": (0, 255, 0),
    "Instrument-offset": (0, 0, 255),
    "Instrument-square-offset": (128, 0, 128),
}

def run_inference_and_get_results(confidence_threshold, img, first_nms_threshold=0.3, second_nms_threshold=0.7):
    model.conf = confidence_threshold
    results = model(img)
    detected_objects = []
    boxes, scores = [], []

    for i in range(len(results.xyxy[0])):
        bbox = results.xyxy[0][i].cpu().numpy()
        class_id = int(bbox[5])
        class_name = model.names[class_id]
        confidence = bbox[4]

        xmin, ymin, xmax, ymax = map(int, bbox[:4].tolist())
        boxes.append([xmin, ymin, xmax, ymax])
        scores.append(confidence.item())
        detected_objects.append({"class": class_name, "confidence": confidence.item(), "bbox": [xmin, ymin, xmax, ymax]})

    if not boxes:
        return []

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    keep_indices_first = nms(boxes_tensor, scores_tensor, first_nms_threshold)
    if not len(keep_indices_first):
        return []

    final_indices = keep_indices_first if second_nms_threshold >= first_nms_threshold else keep_indices_first[nms(boxes_tensor[keep_indices_first], scores_tensor[keep_indices_first], second_nms_threshold)]
    detected_objects_nms = [detected_objects[i] for i in final_indices]
    return detected_objects_nms

def draw_boxes_with_class_colors(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for detection in detections:
        class_name = detection['class']
        confidence = detection['confidence']
        xmin, ymin, xmax, ymax = detection['bbox']
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        text = f"{class_name} {confidence:.2f}"
        draw.text((xmin, ymin), text, fill=color, font=font)
    return image

def crop_detected_areas(image, detections, margin=5):
    cropped_images = []
    for det in detections:
        xmin, ymin, xmax, ymax = det['bbox']
        xmin, ymin = max(0, xmin - margin), max(0, ymin - margin)
        xmax, ymax = min(image.width, xmax + margin), min(image.height, ymax + margin)
        if xmin < xmax and ymin < ymax:
            cropped_images.append(image.crop((xmin, ymin, xmax, ymax)))
    return cropped_images

def enhance_images(images, resize_factor, denoise_strength, denoise_template_window_size, denoise_search_window, thresholding, deskew_angle):
    enhanced_images = []
    for image in images:
        new_size = (int(image.width * resize_factor), int(image.height * resize_factor))
        resized_image = image.resize(new_size, Image.LANCZOS)
        grayscale_image = ImageOps.grayscale(resized_image)
        np_grayscale = np.array(grayscale_image)
        denoised_image = cv2.fastNlMeansDenoising(np_grayscale, None, denoise_strength, denoise_template_window_size, denoise_search_window)
        binarized_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] if thresholding else denoised_image

        coords = np.column_stack(np.where(binarized_image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else (90 - angle) if angle > 45 else -angle
        angle += deskew_angle
        M = cv2.getRotationMatrix2D((binarized_image.shape[1] // 2, binarized_image.shape[0] // 2), angle, 1.0)
        deskewed_image = cv2.warpAffine(binarized_image, M, (binarized_image.shape[1], binarized_image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        sharpened_image = cv2.filter2D(deskewed_image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
        enhanced_images.append(Image.fromarray(sharpened_image))
    return enhanced_images

class RapidOCRTextExtractor:
    def __init__(self, engine):
        self.engine = engine

    def extract_text(self, image_path):
        img = Image.open(image_path)
        open_cv_image = pil_to_cv2(img)
        preprocessed_image = preprocess_for_ocr(open_cv_image)
        result, _ = self.engine(preprocessed_image)
        return ' '.join([res[1] for res in result]) if result else ''

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def preprocess_for_ocr(image, target_size=(300, 300)):
    image = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

ocr_engine = RapidOCR()
text_extractor = RapidOCRTextExtractor(ocr_engine)

# Session state for extracted text data
if 'extracted_data' not in st.session_state:
    st.session_state['extracted_data'] = []

# Function to generate regex pattern
def generate_regex_pattern(parts):
    pattern_map = {
        "System Number": r"(\d{2})",
        "Function Code": r"([A-Z]{2,5})",
        "Loop Sequence": r"(\d{4})"
    }
    selected_patterns = [pattern_map[part] for part in parts]
    return r"\b" + r"\s*".join(selected_patterns) + r"\b"




# Sidebar page selection
page = st.sidebar.selectbox("Select a page", ("Object Detection", "OCR"))

if page == "Object Detection":
    st.write("## Object Detection 🔍\nUse the Object Detection feature to automatically identify and label different instruments and components in your P&ID diagrams. Adjust detection settings as needed.")
    tab_option = st.radio("Select Option", ["Detect Object", "Image Enhancement Tool"], horizontal=True)

    if tab_option == "TensorBoard":
        st.write("## TensorBoard Integration 📊\nView TensorBoard logs and visualize the training metrics and other relevant data.")
        st_tensorboard(logdir=logdir, port=6006, width=1080)

    if tab_option == "Model Performance":
        st.write("## Model Performance Metrics")
        metrics = extract_metrics(logdir)
        display_metrics(metrics)

    if tab_option == "Detect Object":
        dpi = st.number_input("Select DPI for PDF Rendering", min_value=100, max_value=600, value=300, step=50)
        uploaded_files = st.file_uploader("Choose images or PDFs...", type=["jpg", "png", "jpeg", "pdf"], accept_multiple_files=True)
        confidence_threshold = st.slider("Select Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        images = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        images.extend(render_pdf_page_to_png_with_mupdf(uploaded_file, dpi=dpi))
                else:
                    images.append(Image.open(uploaded_file).convert('RGB'))

        if images:
            detect_objects = st.button('Detect Objects')
            if detect_objects:
                for image in images:
                    detections = run_inference_and_get_results(confidence_threshold, image)
                    if detections:
                        image_with_boxes = draw_boxes_with_class_colors(image.copy(), detections)
                        st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)
                        st.session_state['detected_objects'] = detections
                        st.session_state['images_with_boxes'] = image_with_boxes
                    else:
                        st.warning("No objects detected.")

        if 'detected_objects' in st.session_state and st.session_state['detected_objects']:
            margin = st.slider("Select Margin Size for Cropping (pixels)", min_value=0, max_value=50, value=5, step=1, key="crop_margin_slider")
            if st.button('Save Detections'):
                cropped_images_list, cropped_images_paths = [], []
                cropped_images_dir = "cropped_images"
                os.makedirs(cropped_images_dir, exist_ok=True)

                for idx, image in enumerate(images):
                    cropped_images = crop_detected_areas(image, st.session_state['detected_objects'], margin=margin)
                    for crop_idx, cropped_image in enumerate(cropped_images):
                        cropped_image_path = os.path.join(cropped_images_dir, f"cropped_image_{idx}_{crop_idx}.png")
                        cropped_image.save(cropped_image_path)
                        cropped_images_list.append(cropped_image)
                        cropped_images_paths.append(cropped_image_path)

                st.session_state['cropped_images_paths'] = cropped_images_paths
                st.write(f"{len(cropped_images_list)} detected objects.")
                if cropped_images_list:
                    num_to_display = min(len(cropped_images_list), 7)
                    cropped_cols = st.columns(num_to_display)
                    for idx, cropped_col in enumerate(cropped_cols):
                        if idx < len(cropped_images_list):
                            cropped_col.image(cropped_images_list[idx], caption=f'Cropped Object {idx + 1}', width=100)

    if tab_option == "Image Enhancement Tool":
        with st.expander("Improve Detected Objects Quality"):
            if 'cropped_images_paths' in st.session_state and len(st.session_state['cropped_images_paths']) > 0:
                st.success("Detected objects:")
                cols = st.columns(len(st.session_state['cropped_images_paths']))
                for idx, col in enumerate(cols):
                    with col:
                        image = Image.open(st.session_state['cropped_images_paths'][idx])
                        st.image(image, caption=f'Detected Object {idx + 1}', width=100)
                
                col1, col2 = st.columns(2)
                with col1:
                    resize_factor = st.slider("Resize Factor", min_value=0.1, max_value=9.0, value=1.0, step=0.1, key='resize_factor1')
                    denoise_strength = st.slider("Denoise Strength", min_value=0, max_value=200, value=10, key='denoise_strength1')
                    denoise_template_window_size = st.slider("Denoise Template Window Size", min_value=3, max_value=41, value=7, step=2, key='denoise_template_window_size1')
                    denoise_search_window = st.slider("Denoise Search Window", min_value=3, max_value=41, value=21, step=2, key='denoise_search_window1')
                    thresholding = st.checkbox("Thresholding", value=True, key='thresholding1')
                    deskew_angle = st.slider("Deskew Angle", min_value=-90, max_value=90, value=0, key='deskew_angle1')

                    if st.button('Apply Enhancements', key='apply_enhancements1'):
                        enhanced_images = enhance_images(
                            [Image.open(path) for path in st.session_state['cropped_images_paths']],
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

if page == "OCR":
    st.write("## OCR (Optical Character Recognition) 📖\nUse OCR to extract text from detected objects in your images. Customize your extraction with naming conventions and regular expressions for precise analysis.")
    
    # Naming Convention Builder
    st.subheader("Naming Convention Builder")
    st.write("Build your naming convention by selecting the order of components and separators.")
    
    parts = ["None", "System Number", "Function Code", "Loop Sequence"]
    separators = ["-", ""]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        part1 = st.selectbox("First Part", parts, index=2, key="part1")
        if part1 == "System Number":
            system_hint1 = st.text_input("Enter System Number Hint:", value='13', key="system_hint1")
    with col2:
        sep1 = st.selectbox("Separator After First Part", separators, key="sep1")
    with col3:
        part2 = st.selectbox("Second Part", parts, index=2, key="part2")
        if part2 == "System Number":
            system_hint2 = st.text_input("Enter System Number Hint:", value='13', key="system_hint2")
    with col4:
        sep2 = st.selectbox("Separator After Second Part", separators, key="sep2")
    with col5:
        part3 = st.selectbox("Third Part", parts, index=2, key="part3")
        if part3 == "System Number":
            system_hint3 = st.text_input("Enter System Number Hint:", value='13', key="system_hint3")

    selected_parts = [part1, part2, part3]
    selected_separators = [sep1, sep2]

    # Regex patterns for each part
    system_number_pattern = r'\b(\d{2})\b'
    function_code_pattern = r'\b([A-Z]{2,5})\b'
    loop_sequence_pattern = r'\b(\d{4})\b'

    # Extract Text Section
    st.subheader("Extract Text from Cropped Images")

    if st.button('Extract Instruments'):
        extracted_data = []

        if 'cropped_images_paths' in st.session_state:
            for image_path in st.session_state['cropped_images_paths']:
                text = text_extractor.extract_text(image_path)

                # Extract matches for each part
                system_number_matches = re.findall(system_number_pattern, text)
                function_code_matches = re.findall(function_code_pattern, text)
                loop_sequence_matches = re.findall(loop_sequence_pattern, text)

                file_name_match = re.search(r'(.+?)\.png', os.path.basename(image_path))
                drawing_no = file_name_match.group(1) if file_name_match else "Unknown"

                # Assuming we have the same number of function codes and loop sequences
                for function_code, loop_sequence in zip(function_code_matches, loop_sequence_matches):
                    system_number = system_number_matches[0] if system_number_matches else (
                        system_hint1 if part1 == "System Number" else (
                            system_hint2 if part2 == "System Number" else (
                                system_hint3 if part3 == "System Number" else ""
                            )
                        )
                    )

                    parts_dict = {
                        "System Number": system_number,
                        "Function Code": function_code,
                        "Loop Sequence": loop_sequence
                    }

                    tagname_parts = []
                    for i, part in enumerate(selected_parts):
                        if part != "None":
                            tagname_parts.append(parts_dict[part])
                            if i < len(selected_separators):
                                tagname_parts.append(selected_separators[i])
                    tagname = ''.join(tagname_parts).rstrip("-")

                    extracted_data.append({
                        'Tagname': tagname,
                        'Class': 'INSTRUMENT',
                        'System': system_number,
                        'Function_Code': function_code,
                        'Loop_Sequence': loop_sequence,
                        'Drawing_No': drawing_no
                    })

            df = pd.DataFrame(extracted_data)

            col1, col2 = st.columns([1, 1])
            with col1:
                if 'images_with_boxes' in st.session_state and st.session_state['images_with_boxes']:
                    st.image(st.session_state['images_with_boxes'], caption='Uploaded Image with Detections', use_column_width=True)
            with col2:
                if not df.empty:
                    st.write("### Extracted Instruments Data")
                    st.dataframe(df)

                    csv_path = 'output_instruments.csv'
                    df.to_csv(csv_path, sep=';', index=False)

                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name='output_instruments.csv',
                        mime='text/csv'
                    )
                else:
                    st.warning("No matches found or no images to process.")
        else:
            st.warning("No detected objects or cropped images found in the session state.")

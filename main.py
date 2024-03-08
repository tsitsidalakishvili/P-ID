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





st.set_page_config(layout="wide")



# Base directory for your project, relative to the current script file
base_dir = os.path.dirname(__file__)  # Dynamically get the script directory
# base_dir = '.'  # Alternatively, you can use '.' for the current directory

# Relative path to your YOLOv5 directory from the base directory
yolov5_rel_path = 'yolov5'

# Full path to the YOLOv5 directory
yolov5_dir = os.path.join(base_dir, yolov5_rel_path)

# Relative path to your trained model from the base directory
model_rel_path = os.path.join('yolov5', 'runs', 'train', 'exp2', 'weights', 'best.pt')

# Full path to the trained model
model_path = os.path.join(base_dir, model_rel_path)

# Load the trained model with force_reload
model = torch.hub.load(yolov5_dir, 'custom', path=model_path, source='local', force_reload=True)


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

    # Specify a larger font size
    font_size = 20
    try:
        # Use a common font available on most systems
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Default font if specific font is not available
        font = ImageFont.load_default()

    for detection in detections:
        class_name = detection['class']
        bbox = detection['bbox']

        # Check if the class name is in the CLASS_COLORS dictionary
        if class_name in CLASS_COLORS:
            color = CLASS_COLORS[class_name]
        else:
            # Default color if class name not found
            color = (255, 255, 255)

        draw.rectangle(bbox, outline=color, width=2)
        draw.text((bbox[0], bbox[1]), f"{class_name} ({detection['confidence']:.2f})", fill=color, font=font)

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


# Sidebar for navigation
page = st.sidebar.selectbox("Select a page", ("Object Detection", "OCR"))

if page == "Object Detection":
    st.sidebar.write("""
    ## Object Detection 🔍
    Use the Object Detection feature to automatically identify and label different instruments and components in your P&ID diagrams. Adjust detection settings as needed.
    - **Detect Object:** Start object detection on your uploaded image.
    - **Enhancement:** Enhance detected objects for better analysis.
    """)

    st.subheader("Object Detection using YOLOv5")

    # Create radio tabs for "Detect Object" and "Enhancement Parameters"
    tab_option = st.radio("Select Option", ["Detect Object", "Improve Image Quality"], horizontal=True)

  
    # Main code
    if tab_option == "Detect Object":
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            st.session_state['uploaded_image_name'] = uploaded_file.name
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.markdown(f"Uploaded File Name: `{st.session_state['uploaded_image_name']}`")

            if st.button('Detect Objects'):
                detections = run_inference_and_get_results(confidence_threshold, image)
                st.session_state['detected_objects'] = detections

                if detections:
                    st.success(f"Detected Objects: {len(detections)}")
                    image_with_boxes = draw_boxes_with_class_colors(image.copy(), detections)
                    st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)

                    st.markdown("### Class Colors")
                    legend_html = ""
                    for class_name, color in CLASS_COLORS.items():
                        color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
                        legend_html += f"<span style='color: {color_hex};'>{class_name}</span><br>"
                    st.markdown(legend_html, unsafe_allow_html=True)

                    cropped_images_with_classes = crop_detected_areas(image, detections)
                    st.session_state['cropped_images_with_classes'] = cropped_images_with_classes
                else:
                    st.info("No objects detected.")

                # Crop detected areas and store in session state
                st.session_state['cropped_images'] = crop_detected_areas(image, detections)
    
        
    if tab_option == "Improve Image Quality":
        # Check if there are cropped images to display
        if 'cropped_images' in st.session_state and st.session_state['cropped_images']:
            st.success("Detected objects:")
            # Create a multi-column layout to display cropped images
            cols = st.columns(len(st.session_state['cropped_images']))  # Adjust column count as needed
            for idx, col in enumerate(cols):
                with col:
                    st.image(st.session_state['cropped_images'][idx], caption=f'Detected Object {idx + 1}', width=100)  # Adjust width as needed
        else:
            st.warning("No cropped images to display. Please go to the 'Object Detection' page and detect objects first.")

        # Use an expander for enhancement parameters
        with st.expander("Enhancement Parameters", expanded=False):  # Set expanded=True to expand by default
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
                        # Call your enhancement function here
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
                    st.image(st.session_state['enhanced_images'][0], caption='Enhanced Object 1', use_column_width=False)










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


    # Assuming the initial setup remains the same
    
    if selection == "Extract Text":
        st.subheader("Extract Text from Detected Objects")
        default_pattern = r'\b(\d{2})?[,.)]?\s*([A-Z]{2,5})\s*(\d{4})(?:\s*[A-Z])?\b'
        regex_pattern = st.text_input("Enter the custom regex pattern for instrument numbers:", value=default_pattern)
        system_number_hint = st.text_input("System Number Hint:", key="system_number_hint")

        system_number_position = st.radio(
            "Select where to place the System Number:",
            ('Beginning', 'End'), 
            index=0, 
            key="system_number_position"
        )

        if st.button('Extract Instruments'):
            extracted_data = []

            # Ensure you've processed and stored cropped images with class names in st.session_state['cropped_images_with_classes'] beforehand
            for idx, detected_object in enumerate(st.session_state['detected_objects']):
                class_name = detected_object['class']  # Extract the class name
                image = st.session_state['cropped_images_with_classes'][idx]  # Accessing the cropped image directly
                
                # Here, you would perform OCR on each cropped image
                text = extract_text_from_image(image)
                instrument_pattern = re.compile(regex_pattern)
                matches = instrument_pattern.findall(text)

                for match in matches:
                    system_number = match[0].strip() if match[0].strip() else system_number_hint
                    function_code = match[1].strip()
                    loop_sequence = match[2].strip()

                    if system_number_position == 'Beginning':
                        tagname = f"{system_number}-{function_code}-{loop_sequence}"
                    else:
                        tagname = f"{function_code}-{loop_sequence}-{system_number}"

                    extracted_data.append({
                        "Index": len(extracted_data) + 1,
                        "Original Number": ''.join(match).strip(),
                        "TAGNAME": tagname,
                        "SYSTEM": system_number,
                        "FUNCTION_CODE": function_code,
                        "LOOP SEQUENCE": loop_sequence,
                        "DRAWING_NO": st.session_state.get('uploaded_image_name', 'No File Uploaded'),
                        "CLASS": class_name  # Including the class name
                    })

            st.session_state['extracted_data'] = extracted_data

        # Continue with displaying and editing extracted instrument numbers...

        # Displaying and editing the extracted instrument numbers
        if 'extracted_data' in st.session_state and st.session_state['extracted_data']:
            for data in st.session_state['extracted_data']:
                # Use .get() to safely access 'TAGNAME' with a default value if not found
                edited_tagname = st.text_input(
                    f"Edit Tagname {data['Index']}",
                    value=data.get('TAGNAME', 'N/A'),  # Default value 'N/A' if 'TAGNAME' not found
                    key=f"edit_tag_{data['Index']}"
                )
                # Here you can update the dictionary with edited_tagname if needed

            if st.button("Confirm Edits"):
                # Optionally process the confirmed edits here
                st.success("Instrument numbers and Tagnames updated.")

        # Example adjustment for displaying the table with a "CLASS" column
        if st.button("Show Table"):
            if 'extracted_data' in st.session_state and st.session_state['extracted_data']:
                # Convert the extracted data to a DataFrame
                df = pd.DataFrame(st.session_state['extracted_data'])

                # Specify the desired column order, ensuring CLASS follows TAGNAME
                column_order = ["Index", "Original Number", "TAGNAME", "CLASS", "SYSTEM", "FUNCTION_CODE", "LOOP SEQUENCE", "DRAWING_NO"]

                # Reorder the DataFrame according to the specified column order
                # Note: This step assumes all the specified columns exist in your DataFrame.
                # If there are any additional or missing columns, adjust the column_order list accordingly.
                df = df[column_order]

                # Display the reordered DataFrame
                st.dataframe(df)
            else:
                st.warning("No data available. Please extract and edit instrument numbers and Tagnames first.")

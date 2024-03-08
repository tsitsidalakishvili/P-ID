from PIL import Image, ImageDraw
import random
import imageio

def generate_random_detections_gif(pid_diagram_path, gif_path, num_detections=50):
    pid_image = Image.open(pid_diagram_path)
    img_width, img_height = pid_image.size

    class_map = {
        '0': 'Instrument',
        '1': 'Instrument-square',
        '2': 'Instrument-offset',
        '3': 'Instrument-square-offset'
    }

    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan']

    detected_instruments = []
    for _ in range(num_detections):
        x1, y1 = random.randint(0, img_width - 100), random.randint(0, img_height - 100)
        x2, y2 = x1 + random.randint(50, 150), y1 + random.randint(50, 150)
        class_id = str(random.choice(list(class_map.keys())))
        color = random.choice(colors)
        detected_instruments.append({
            'bbox': (x1, y1, x2, y2),
            'class_id': class_id,
            'label': f"{class_map[class_id]}-{random.randint(100, 999)}",
            'color': color
        })

    progress_images = []
    for instrument in detected_instruments:
        draw = ImageDraw.Draw(pid_image)
        bbox = instrument['bbox']
        class_id = instrument['class_id']
        class_label = class_map[class_id]
        label = f"{class_label}: {instrument['label']}"
        color = instrument['color']
        
        draw.rectangle(bbox, outline=color, width=2)
        draw.text(bbox[:2], label, fill=color)
        progress_images.append(pid_image.copy())

    progress_images.append(pid_image)
    imageio.mimsave(gif_path, progress_images, duration=0.9)

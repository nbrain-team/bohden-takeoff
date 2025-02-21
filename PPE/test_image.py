import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load trained YOLOv8 model
model = YOLO("models/best_2.pt") 

# Path to the test image
image_path = "D:/L and T/test.jpg"  # Update this path to your test image

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Run YOLO model on GPU
results = model(image, device=0)  # Use GPU

# Draw bounding boxes and labels
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = box.cls[0]
        confidence = box.conf[0]
        label_text = f"{model.names[int(label)]}: {confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw background rectangle for text
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# Save the image with bounding boxes and labels to disk
output_path = "output_image.jpg"
cv2.imwrite(output_path, image)

# Display the image using Matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("PPE Detection")
plt.axis('off')
plt.show()

print(f"Detection complete. Image saved to {output_path}")

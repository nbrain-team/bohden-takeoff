import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load trained YOLOv8 model
model = YOLO("models/best_2.pt")  # Correct path to the trained model

cap = cv2.VideoCapture(0)  # Open webcam

# Ensure the output directory exists
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on GPU
    results = model(frame, device=0)  # Use GPU

    # Draw bounding boxes and labels
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = box.cls[0]
            confidence = box.conf[0]
            label_text = f"{model.names[int(label)]}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame with bounding boxes and labels to disk
    frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, frame)
    frame_count += 1

    # Display the frame using Matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("PPE Detection")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()

cap.release()
cv2.destroyAllWindows()

print(f"Detection complete. Frames saved to {output_dir}")

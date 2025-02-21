from ultralytics import YOLO
import os

def infer_image(image_path):
    # Load the trained model
    model = YOLO("blueprint.pt")

    # Specify the directory to save the annotated images
    save_dir = os.path.join(os.getcwd(), "runs", "detect", "predict")
    os.makedirs(save_dir, exist_ok=True)

    # Perform inference and save the results in the specified directory
    results = model.predict(image_path, save=True, save_dir=save_dir)

    # Print the results
    print(f"Inference results for {image_path}:")
    for result in results:
        print(f"Detected {len(result.boxes)} objects:")
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            confidence = float(box.conf)  # Convert tensor to float
            bbox = box.xyxy.tolist()  # Convert tensor to list
            print(f"Class: {class_name}, Confidence: {confidence:.2f}, Bounding Box: {bbox}")

if __name__ == '__main__':
    # Specify the path to the test image
    image_path = os.path.join(os.getcwd(), "test.jpg")
    infer_image(image_path)

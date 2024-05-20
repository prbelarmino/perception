import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101

# Initialize the webcam (assuming the webcam index is 0)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the pre-trained DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True).eval()

# Define the image transformation
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define colors for the segmentation mask
COLORS = np.random.uniform(0, 255, size=(21, 3))

# Define class names for COCO dataset (21 classes for the pre-trained model)
CLASS_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train',
    'tv/monitor'
]

def create_legend(classes_present):
    num_classes = len(classes_present)
    legend_height = max(300, num_classes * 25)
    legend = np.zeros((legend_height, 200, 3), dtype=np.uint8)
    for i, label in enumerate(classes_present):
        class_name = CLASS_NAMES[label]
        color = COLORS[label]
        cv2.putText(legend, class_name, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.rectangle(legend, (150, 10 + i * 25), (170, 30 + i * 25), color, -1)
    return legend

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Apply the transformation to the frame
    input_image = transform(frame).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0)

    # Create a segmentation mask
    mask = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)
    classes_present = np.unique(output_predictions)
    for label in classes_present:
        mask[output_predictions == label] = COLORS[label]

    # Combine the original image with the segmentation mask
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    segmented_image = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)

    # Create the legend only with the present classes
    legend = create_legend(classes_present)

    # Display the original, segmented images and the legend
    cv2.imshow('Original Image', frame)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.imshow('Legend', legend)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

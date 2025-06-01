import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture("/dev/video2")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

def process_thermal_frame(frame):
    """Process raw YUYV frame for thermal segmentation"""
    # Convert BGR back to YUYV (if OpenCV auto-converted it)
    if frame.shape[2] == 3:  # BGR (3-channel)
        yuyv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuyv[:, :, 0]  # Extract Y (luma) channel
    else:  # Assume already YUYV (2-channel)
        y_channel = frame[:, :, 0]  # First channel is Y

    # Normalize and enhance contrast
    normalized = cv2.normalize(
        y_channel, None, 0, 255, 
        cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        normalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Denoise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cleaned

while True:
    ret, frame = cap.read()
    if not ret:
        break

    segmented = process_thermal_frame(frame)

    # Find contours
    contours, _ = cv2.findContours(
        segmented,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw bounding boxes
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow("Raw Thermal", frame)
    cv2.imshow("Segmented", segmented)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
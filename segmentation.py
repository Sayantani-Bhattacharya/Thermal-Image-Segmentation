import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initialize camera
cap = cv2.VideoCapture("/dev/video2")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

# Initialize background subtractor for motion tracking
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)


def process_thermal_frame(frame):
    """Process raw YUYV frame for thermal segmentation"""
    # # Convert BGR back to YUYV (if OpenCV auto-converted it)
    # if frame.shape[2] == 3:  # BGR (3-channel)
    #     yuyv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #     y_channel = yuyv[:, :, 0]  # Extract Y (luma) channel
    # else:  # Assume already YUYV (2-channel)
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

    # Denoise: Morphological opening (Erode followed by Dilate)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned

def draw_heat_map(frame):
    """Draw a heat map on the thermal frame"""
    # Convert to grayscale if not already
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Normalize for heat map
    heat_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heat_map

# Initialize variables for motion tracking
previous_frame = None
# List to store centroids of the largest motion segment
centroids = []


# Initialize matplotlib for live plotting
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 6))
x_coords, y_coords = [], []  # Lists to store x and y coordinates of centroids
line, = ax.plot([], [], marker='o', linestyle='-', color='b', label='Centroid Path')
ax.set_title("Centroid Path of Largest Motion Segment")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend()
ax.grid()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    segmented = process_thermal_frame(frame)   

    # Find contours.
    contours, _ = cv2.findContours(
        segmented,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw bounding boxes for segments.
    for cnt in contours:
        if cv2.contourArea(cnt) > 100: #50
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Motion tracking using background subtraction
    fg_mask = background_subtractor.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Denoise motion mask

    # Find contours in the motion mask
    motion_contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

        # Find the largest motion contour and calculate its centroid
    if motion_contours:
        largest_contour = max(motion_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Filter small motion areas
            # Calculate centroid
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:  # Avoid division by zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))  # Store the centroid
                # Draw the centroid on the frame
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red dot for centroid
                # Update the live plot
                line.set_xdata(cx)
                line.set_ydata(cy)
                ax.set_xlim(0, 256)  # Set x-axis limits based on frame width
                ax.set_ylim(0, 192)  # Set y-axis limits based on frame height
                plt.draw()
                plt.pause(0.001)  # Pause briefly to update the plot

    # # Draw bounding boxes for multiple moving objects
    # for cnt in motion_contours:
    #     if cv2.contourArea(cnt) > 100:  # Filter small motion areas
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)  # Blue for motion

    # Draw bounding box for only the largest segment.
    if centroids:
        largest_centroid = centroids[-1]  # Get the last centroid
        cv2.rectangle(frame, (largest_centroid[0]-5, largest_centroid[1]-5), (largest_centroid[0]+5, largest_centroid[1]+5), (0, 255, 255), 2)  # Yellow box for largest segment


    resized_frame = cv2.resize(frame, (800, 600))
    resized_segmented = cv2.resize(segmented, (800, 600))
    heat_map_frame = draw_heat_map(resized_frame)
    resized_hestMap = cv2.resize(heat_map_frame, (800, 600))
    resized_fg_mask = cv2.resize(fg_mask, (800, 600))


    cv2.imshow("Raw Thermal", resized_frame)
    cv2.imshow("Segmented", resized_segmented)
    cv2.imshow("Heat Map", resized_hestMap)
    cv2.imshow("Motion Mask", resized_fg_mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
plt.savefig("segment_trajectory.png")  # Save the final plot

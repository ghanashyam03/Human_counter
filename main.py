import cv2
from object_tracking import ObjectTracker
from yolo_detection import YOLODetector

# Create instances of the ObjectTracker and YOLODetector
object_tracker = ObjectTracker()
yolo_detector = YOLODetector()

# Open a connection to the webcam (0 represents the default webcam)
cap = cv2.VideoCapture(0)

# Dictionary to store track IDs and corresponding class IDs
track_dict = {}

# Counter for the total number of people detected
total_people_detected = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    detected_objects = yolo_detector.detect_people(frame)
    print("Detected Objects:", detected_objects)  # Debugging line to inspect detected objects

    # Object tracking
    tracked_objects = object_tracker.track_objects(detected_objects)
    print("Tracked Objects:", tracked_objects)  # Debugging line to inspect tracked objects

    for obj in tracked_objects:
        x, y, x2, y2, track_id, class_id = obj  # Extract class_id from tracked objects
        center_x = (x + x2) // 2
        center_y = (y + y2) // 2

        # Check if this person is already detected
        is_new_person = True
        for tracked_id, tracked_class_id in track_dict.items():
            # Unpack the tracked ID
            tracked_center_x, tracked_center_y, tracked_class = tracked_id

            # Calculate the distance between the centers
            distance = ((center_x - tracked_center_x) ** 2 + (center_y - tracked_center_y) ** 2) ** 0.5

            # If the distance is below a threshold and the class_id matches, consider it the same person
            if distance < 50 and class_id == tracked_class:
                is_new_person = False
                break

        if is_new_person:
            track_dict[(center_x, center_y, class_id)] = class_id
            total_people_detected += 1

            # Draw a bounding box around the detected person
            cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {class_id:.2f}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the total number of people detected at the top of the frame
    cv2.putText(frame, f"Total People Detected: {total_people_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Live Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

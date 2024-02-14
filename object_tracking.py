from sort import Sort
import numpy as np

class ObjectTracker:
    def __init__(self):
        self.tracker = Sort()
        self.track_dict = {}

    def track_objects(self, detected_objects):
        # Check if detected_objects is empty or malformed
        if not detected_objects:
            return []

        # Convert detected bounding boxes to a format compatible with SORT
        trackers = np.array([[obj[0], obj[1], obj[2], obj[3], obj[4]] for obj in detected_objects])

        # Update the SORT tracker
        tracked_objects = self.tracker.update(trackers)

        # Extend the tracked objects to include class_id
        tracked_objects_with_class = []
        for obj in tracked_objects:
            x, y, x2, y2, track_id = obj
            if 0 <= int(track_id) < len(detected_objects):  # Check if track_id is within the valid range
                class_id = int(detected_objects[int(track_id)][4])  # Extract class_id (confidence) from detected_objects
                tracked_objects_with_class.append((x, y, x2, y2, track_id, class_id))

        return tracked_objects_with_class

    # Other methods related to object tracking

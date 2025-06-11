import unittest
import sys
import os

# Add the project root to the Python path to allow importing 'main'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from main import VehicleTracker # Assuming VehicleTracker is in main.py

class TestVehicleTracker(unittest.TestCase):

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)

    def setUp(self):
        """Set up for each test method."""
        self.tracker = VehicleTracker()
        # Reset global MAX_DISAPPEARED_FRAMES for tests, original value in main.py is 30
        # This allows tests to run faster.
        # Note: This directly modifies the global in the imported main module for the scope of these tests.
        # A better way would be to pass config to VehicleTracker or make MAX_DISAPPEARED_FRAMES a class/instance var.
        main.MAX_DISAPPEARED_FRAMES = 3
        main.MAX_TRACKING_DISTANCE = 50 # Default is 75, use 50 for tests

    def test_01_register_single_object(self):
        """Test registering a single object."""
        detections = [(([10, 10, 30, 30], "car", 0.9),)] # One detection
        self.tracker.update(detections[0], frame_num=1)

        self.assertEqual(len(self.tracker.tracked_objects), 1)
        self.assertTrue(0 in self.tracker.tracked_objects) # ID 0 should be assigned
        self.assertEqual(self.tracker.tracked_objects[0]["bbox"], [10, 10, 30, 30])
        self.assertEqual(self.tracker.tracked_objects[0]["centroid"], self._get_centroid([10,10,30,30]))

    def test_02_register_multiple_objects(self):
        """Test registering multiple objects simultaneously."""
        detections = [
            (([10, 10, 30, 30], "car", 0.9),),
            (([50, 50, 70, 70], "truck", 0.8),)
        ]
        # Simulate two frames to register them distinctly if update processes one list
        # Or pass as a list of detections if update handles multiple new items
        self.tracker.update([detections[0][0]], frame_num=1)
        self.tracker.update([detections[1][0]], frame_num=1) # Same frame for simultaneous

        self.assertEqual(len(self.tracker.tracked_objects), 2)
        self.assertTrue(0 in self.tracker.tracked_objects)
        self.assertTrue(1 in self.tracker.tracked_objects)
        self.assertEqual(self.tracker.tracked_objects[1]["bbox"], [50, 50, 70, 70])

    def test_03_update_existing_object(self):
        """Test updating an existing object's position."""
        initial_bbox = ([10, 10, 30, 30], "car", 0.9)
        self.tracker.update([initial_bbox], frame_num=1)
        self.assertEqual(self.tracker.tracked_objects[0]["consecutive_hits"], 1)


        updated_bbox = ([12, 12, 32, 32], "car", 0.92) # Same object, moved slightly
        self.tracker.update([updated_bbox], frame_num=2)

        self.assertEqual(len(self.tracker.tracked_objects), 1) # Still one object
        self.assertTrue(0 in self.tracker.tracked_objects)
        self.assertEqual(self.tracker.tracked_objects[0]["bbox"], [12, 12, 32, 32])
        self.assertEqual(self.tracker.tracked_objects[0]["centroid"], self._get_centroid([12,12,32,32]))
        self.assertIsNotNone(self.tracker.tracked_objects[0]["prev_centroid"])
        self.assertEqual(self.tracker.tracked_objects[0]["prev_centroid"], self._get_centroid([10,10,30,30]))
        self.assertEqual(self.tracker.tracked_objects[0]["consecutive_hits"], 2)


    def test_04_deregister_disappeared_object(self):
        """Test that an object is deregistered after disappearing for MAX_DISAPPEARED_FRAMES."""
        main.MAX_DISAPPEARED_FRAMES = 3 # ensure it's set for this test
        detection1 = [(([10, 10, 30, 30], "car", 0.9),)]
        self.tracker.update(detection1[0], frame_num=1)
        self.assertEqual(len(self.tracker.tracked_objects), 1)
        obj_id = 0 # First object

        # Simulate frames where the object is not detected
        for i in range(main.MAX_DISAPPEARED_FRAMES):
            self.tracker.update([], frame_num=2 + i) # Empty detections
            self.assertTrue(obj_id in self.tracker.tracked_objects, f"Should still be tracked at frame {2+i}")
            self.assertEqual(self.tracker.disappeared_frames[obj_id], i + 1)

        # One more frame, object should be deregistered
        self.tracker.update([], frame_num=2 + main.MAX_DISAPPEARED_FRAMES)
        self.assertFalse(obj_id in self.tracker.tracked_objects, "Object should be deregistered")
        self.assertEqual(len(self.tracker.tracked_objects), 0)


    def test_05_object_reappears_before_deregister(self):
        """Test an object reappearing before being deregistered."""
        main.MAX_DISAPPEARED_FRAMES = 3
        bbox1 = ([10, 10, 30, 30], "car", 0.9)
        self.tracker.update([bbox1], frame_num=1)
        obj_id = 0

        # Disappears for 2 frames (less than MAX_DISAPPEARED_FRAMES)
        self.tracker.update([], frame_num=2)
        self.tracker.update([], frame_num=3)
        self.assertTrue(obj_id in self.tracker.tracked_objects)
        self.assertEqual(self.tracker.disappeared_frames[obj_id], 2)
        self.assertEqual(self.tracker.tracked_objects[obj_id]["consecutive_hits"], 0)


        # Reappears
        bbox2 = ([15, 15, 35, 35], "car", 0.9)
        self.tracker.update([bbox2], frame_num=4)

        self.assertTrue(obj_id in self.tracker.tracked_objects)
        self.assertEqual(len(self.tracker.tracked_objects), 1)
        self.assertEqual(self.tracker.disappeared_frames[obj_id], 0) # Reset
        self.assertEqual(self.tracker.tracked_objects[obj_id]["bbox"], [15,15,35,35])
        self.assertEqual(self.tracker.tracked_objects[obj_id]["consecutive_hits"], 1) # Reset on new detection streak


    def test_06_new_object_while_tracking_another(self):
        """Test adding a new object while another is being tracked."""
        bbox_car1_f1 = ([10, 10, 30, 30], "car", 0.9)
        self.tracker.update([bbox_car1_f1], frame_num=1) # Car 1 (ID 0) appears

        bbox_car1_f2 = ([12, 12, 32, 32], "car", 0.9)
        bbox_truck1_f2 = ([100, 100, 150, 150], "truck", 0.8)
        self.tracker.update([bbox_car1_f2, bbox_truck1_f2], frame_num=2) # Car 1 moves, Truck 1 (ID 1) appears

        self.assertEqual(len(self.tracker.tracked_objects), 2)
        self.assertTrue(0 in self.tracker.tracked_objects) # Car 1
        self.assertTrue(1 in self.tracker.tracked_objects) # Truck 1
        self.assertEqual(self.tracker.tracked_objects[0]["bbox"], [12,12,32,32])
        self.assertEqual(self.tracker.tracked_objects[1]["bbox"], [100,100,150,150])

    def test_07_max_tracking_distance(self):
        """Test that an object moving too far is considered a new object."""
        main.MAX_TRACKING_DISTANCE = 50 # Ensure for test
        bbox_obj1_f1 = ([10, 10, 30, 30], "car", 0.9)
        self.tracker.update([bbox_obj1_f1], frame_num=1) # Obj1 (ID 0)

        # Object moves further than MAX_TRACKING_DISTANCE
        # Centroid of obj1_f1 is (20,20)
        # New detection centroid is (20+50+1, 20) = (71,20)
        # Distance is 51, which is > 50
        bbox_obj2_f2 = ([10 + 51, 10, 30 + 51, 30], "car", 0.9)
        self.tracker.update([bbox_obj2_f2], frame_num=2)

        self.assertEqual(len(self.tracker.tracked_objects), 2) # Should be two objects now
        self.assertTrue(0 in self.tracker.tracked_objects) # Original object (ID 0), now disappeared=1
        self.assertTrue(1 in self.tracker.tracked_objects) # New object (ID 1)
        self.assertEqual(self.tracker.disappeared_frames[0], 1)
        self.assertEqual(self.tracker.tracked_objects[1]["bbox"], bbox_obj2_f2[0])

    def test_08_no_detections_update(self):
        """Test updating with no detections when objects are already tracked."""
        self.tracker.update([(([10,10,20,20], "car", 0.9),)], frame_num=1)
        self.tracker.update([(([50,50,60,60], "bus", 0.9),)], frame_num=1)
        self.assertEqual(len(self.tracker.tracked_objects), 2)

        self.tracker.update([], frame_num=2) # No detections
        self.assertEqual(len(self.tracker.tracked_objects), 2) # Still tracked
        self.assertEqual(self.tracker.disappeared_frames[0], 1)
        self.assertEqual(self.tracker.disappeared_frames[1], 1)
        self.assertEqual(self.tracker.tracked_objects[0]["consecutive_hits"], 0)


if __name__ == '__main__':
    unittest.main()

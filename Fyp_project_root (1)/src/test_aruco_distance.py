import cv2
import numpy as np
import math

def test_aruco_distance():
    print("Initializing camera for ArUco distance test...")
    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set up ArUco dictionary and parameters
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        
        # For newer OpenCV versions (>= 4.7)
        if hasattr(cv2.aruco, 'ArucoDetector'):
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            use_detector = True
        else:
            use_detector = False
    except AttributeError as e:
        print(f"Error initializing ArUco: {e}")
        print("Make sure you have 'opencv-contrib-python' installed.")
        return

    print("\n[ArUco Calibration Test]")
    print("Ensure both Marker 0 (80cm) and Marker 1 (180cm) are visible in the frame.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        if use_detector:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        marker_centers = {}

        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Find centers of markers 0 and 1
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in [0, 1]:
                    # Calculate center
                    c = corners[i][0]
                    center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
                    center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
                    marker_centers[marker_id] = (center_x, center_y)

                    # Draw center point and ID
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID {marker_id}", (center_x + 10, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate distance if both markers are detected
        if 0 in marker_centers and 1 in marker_centers:
            pt0 = marker_centers[0]
            pt1 = marker_centers[1]

            # Draw line between them
            cv2.line(frame, pt0, pt1, (255, 0, 0), 2)

            # Euclidean distance in pixels
            pixel_distance = math.sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
            
            # Since the markers are exactly 100 cm (1 meter) apart center-to-center
            # We can calculate the pixel-to-meter ratio
            pixels_per_meter = pixel_distance / 1.0  # 1.0 meters (100cm)
            
            # Draw distance info
            info_text1 = f"Pixel Dist: {pixel_distance:.1f} px"
            info_text2 = f"Ratio: {pixels_per_meter:.1f} px/m"
            
            # Display info
            cv2.putText(frame, "BOTH MARKERS DETECTED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, info_text1, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, info_text2, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Calculate what height marker 0 represents based on current ratio
            # To test the scale, we know marker 0 is at 0.8m
            # If a person's head was at pt_top (y_top) and feet at pt_bottom (y_bottom), we could use this ratio.
        else:
            cv2.putText(frame, "Waiting for both markers (ID 0 & 1)...", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("ArUco Distance Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_aruco_distance()

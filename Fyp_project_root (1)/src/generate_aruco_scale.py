"""
generate_aruco_scale.py
-----------------------
Generates a printable height-measurement scale with two ArUco markers.

The printed poster should be mounted on a wall. The two ArUco markers
serve as reference points at KNOWN heights from the floor.

Default setup:
  - Bottom marker (ID 0): mount at 80cm from floor
  - Top marker (ID 1): mount at 180cm from floor
  - Height markings every 5cm between them

Print this at the largest size your printer supports (A3 ideally, A4 works).
The ACTUAL marker positions don't need to match the printed markings —
you just need to measure the real distance between the two markers
after mounting and input that into the software.

Usage:
    python generate_aruco_scale.py
"""

import numpy as np
import cv2
import os


def generate_aruco_scale(
    output_path="aruco_height_scale.png",
    img_width=900,       # pixels wide
    img_height=3600,     # pixels tall (4:1 aspect for a tall banner)
    marker_size_px=200,  # ArUco marker size in pixels
    bottom_cm=80,        # Bottom marker height label
    top_cm=180,          # Top marker height label
):
    """
    Generate a tall printable PNG with:
    - Two ArUco markers (ID 0 at bottom, ID 1 at top)
    - Height markings every 5cm between bottom_cm and top_cm
    - Clear labels and instructions
    """
    # Create white canvas
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Generate ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    marker_bottom = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_size_px)
    marker_top = cv2.aruco.generateImageMarker(aruco_dict, 1, marker_size_px)

    # Convert markers to 3-channel
    marker_bottom_3ch = cv2.cvtColor(marker_bottom, cv2.COLOR_GRAY2BGR)
    marker_top_3ch = cv2.cvtColor(marker_top, cv2.COLOR_GRAY2BGR)

    # Layout: Center the markers and scale markings
    # The total width of markers + scale + text is roughly 460px.
    margin_left = (img_width - 460) // 2
    scale_x = margin_left + marker_size_px + 60  # where scale line starts
    scale_x_end = scale_x + 80                    # where scale ticks end

    # Vertical positions for markers (in image pixels)
    # Leave margins at top and bottom
    top_margin = 200
    bottom_margin = 200
    usable_height = img_height - top_margin - bottom_margin

    # Bottom marker position (in image: lower = higher y value)
    bottom_marker_y = img_height - bottom_margin - marker_size_px
    top_marker_y = top_margin

    # Place bottom marker (ID 0)
    y1_b = bottom_marker_y
    y2_b = y1_b + marker_size_px
    x1_b = margin_left
    x2_b = x1_b + marker_size_px
    img[y1_b:y2_b, x1_b:x2_b] = marker_bottom_3ch

    # Place top marker (ID 1)
    y1_t = top_marker_y
    y2_t = y1_t + marker_size_px
    x1_t = margin_left
    x2_t = x1_t + marker_size_px
    img[y1_t:y2_t, x1_t:x2_t] = marker_top_3ch

    # Marker center positions (for the scale line)
    bottom_center_y = (y1_b + y2_b) // 2
    top_center_y = (y1_t + y2_t) // 2

    # Draw vertical scale line
    cv2.line(img, (scale_x, top_center_y), (scale_x, bottom_center_y), (0, 0, 0), 3)

    # Draw height markings every 5cm
    total_cm = top_cm - bottom_cm  # e.g., 100cm
    px_per_cm_scale = (bottom_center_y - top_center_y) / total_cm

    for cm in range(bottom_cm, top_cm + 1, 5):
        # Position in image pixels (bottom = higher y)
        frac = (cm - bottom_cm) / total_cm
        y_pos = int(bottom_center_y - frac * (bottom_center_y - top_center_y))

        # Major tick every 10cm, minor every 5cm
        is_major = (cm % 10 == 0)
        tick_len = 60 if is_major else 30
        thickness = 3 if is_major else 2
        color = (0, 0, 0) if is_major else (100, 100, 100)

        # Draw tick mark
        cv2.line(img, (scale_x - 10, y_pos), (scale_x + tick_len, y_pos), color, thickness)

        # Label for major ticks
        if is_major:
            label = f"{cm} cm"
            font_scale = 1.2
            cv2.putText(img, label, (scale_x + tick_len + 15, y_pos + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)

    # --- Labels for markers ---
    # Bottom marker label
    cv2.putText(img, f"MARKER 0", (x1_b, y2_b + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    cv2.putText(img, f"Mount at {bottom_cm}cm", (x1_b, y2_b + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)

    # Top marker label
    cv2.putText(img, f"MARKER 1", (x1_t, y1_t - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    cv2.putText(img, f"Mount at {top_cm}cm", (x1_t, y1_t - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)

    # --- Title and instructions ---
    cv2.putText(img, "BMI ESTIMATION", (img_width // 2 - 220, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    cv2.putText(img, "HEIGHT SCALE", (img_width // 2 - 180, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (50, 50, 50), 3)

    # Instructions at bottom
    instructions = [
        "SETUP INSTRUCTIONS:",
        "1. Print this page as large as possible",
        "2. Mount on wall with MARKER 0 center at 80cm from floor",
        "3. Mount MARKER 1 center at 180cm from floor",
        "4. Measure actual distance between marker centers",
        "5. Person stands NEXT TO the scale, facing camera",
    ]
    y_inst = img_height - 40
    for line in reversed(instructions):
        color = (0, 0, 180) if "INSTRUCTIONS" in line else (60, 60, 60)
        thickness = 2 if "INSTRUCTIONS" in line else 1
        cv2.putText(img, line, (margin_left - 20, y_inst),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        y_inst -= 35

    # --- Border ---
    cv2.rectangle(img, (5, 5), (img_width - 5, img_height - 5), (0, 0, 0), 3)

    # Draw cut lines (dashed) at corners
    for corner_x in [20, img_width - 20]:
        for corner_y in [20, img_height - 20]:
            cv2.drawMarker(img, (corner_x, corner_y), (150, 150, 150),
                          cv2.MARKER_CROSS, 20, 1)

    cv2.imwrite(output_path, img)
    print(f"\n[SUCCESS] ArUco height scale saved to: {output_path}")
    print(f"  Image size: {img_width} x {img_height} px")
    print(f"  Bottom marker (ID 0): label = {bottom_cm}cm")
    print(f"  Top marker (ID 1): label = {top_cm}cm")
    print(f"  Scale range: {bottom_cm}cm - {top_cm}cm")
    print(f"\nPRINT INSTRUCTIONS:")
    print(f"  1. Print this image as large as possible (A3 paper or poster)")
    print(f"  2. Cut out and mount on wall/board")
    print(f"  3. Position MARKER 0 center at exactly {bottom_cm}cm from floor")
    print(f"  4. Position MARKER 1 center at exactly {top_cm}cm from floor")
    print(f"  5. Measure the actual center-to-center distance between markers")
    print(f"     (should be {top_cm - bottom_cm}cm if mounted correctly)")

    return output_path


if __name__ == "__main__":
    # Also generate individual markers for easier printing
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Individual marker 0 (bottom, 80cm)
    m0 = cv2.aruco.generateImageMarker(aruco_dict, 0, 500)
    cv2.imwrite("aruco_marker_0_bottom.png", m0)
    print("Saved individual marker: aruco_marker_0_bottom.png (mount at 80cm)")

    # Individual marker 1 (top, 180cm)
    m1 = cv2.aruco.generateImageMarker(aruco_dict, 1, 500)
    cv2.imwrite("aruco_marker_1_top.png", m1)
    print("Saved individual marker: aruco_marker_1_top.png (mount at 180cm)")

    # Full scale
    generate_aruco_scale()

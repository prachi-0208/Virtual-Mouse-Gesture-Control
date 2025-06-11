import numpy as np

# Function to calculate the angle between three points (in degrees)
# Used to identify finger bending or extension by measuring joint angles
def get_angle(a, b, c):
    """
    Parameters:
    - a, b, c: Tuples (x, y) representing coordinates of three landmarks
               where 'b' is the center joint, and 'a' and 'c' are adjacent joints.

    Returns:
    - angle in degrees between the vectors ab and bc.
    """
    # Calculate the angle using arctangent difference formula
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    
    # Convert radians to degrees and take absolute value
    angle = np.abs(np.degrees(radians))
    
    # Return the calculated angle
    return angle

# Function to calculate the Euclidean distance between two landmarks
# Used to detect proximity of thumb to another finger (e.g., for clicks/screenshot)
def get_distance(landmark_list):
    """
    Parameters:
    - landmark_list: List containing two landmark tuples [(x1, y1), (x2, y2)]

    Returns:
    - Distance value (scaled from 0 to 1000 for easier gesture thresholding)
    """
    if len(landmark_list) < 2:
        return 0  # Not enough points to measure distance

    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]

    # Compute Euclidean distance
    L = np.hypot(x2 - x1, y2 - y1)

    # Scale distance to a more usable range (0 to 1000)
    return np.interp(L, [0, 1], [0, 1000])

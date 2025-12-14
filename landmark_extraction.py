import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os
import csv
import pandas as pd


#====Landmarks identificators
L_SH, R_SH = 11, 12 # shoulders
L_HP, R_HP = 23, 24 # hips
L_ELB, R_ELB = 13, 14 # elbows
L_WR, R_WR = 15, 16 # wrists
L_KNEE, R_KNEE = 25, 26 # knees
L_ANK, R_ANK = 27, 28 # ankles
L_FT, R_FT = 31, 32 # feet
HEAD = 0  # head
TORSO = 12 # torso
    
    
def interpolate_landmarks(landmarks_sequence):
    """
    Interpolates missing landmarks (NaN) over time for each landmark.
    landmarks_sequence : np.ndarray of shape (n_frames, 33, 3)
    """
    arr = landmarks_sequence.copy()
    n_frames, n_landmarks, _ = arr.shape

    for j in range(n_landmarks):  # pour chaque landmark
        for k in range(3):  # x, y, z
            values = arr[:, j, k]
            mask = np.isnan(values)
            if np.any(~mask):
                valid = np.where(~mask)[0]
                arr[:, j, k] = np.interp(np.arange(n_frames), valid, values[valid])
    return arr



def extract_pose_from_video_interpolated(filename, show = False): 
    """
    Detect pose landmarks from a video using MediaPipe.
    Missing landmarks are interpolated.
    Optionally visualize the interpolated landmarks.
    """
    
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise ValueError("Unable to open the video file.")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # For the tqdm progress bar
    all_landmarks = []
    frames = []

    # === Video lecture with progress bar ===
    for _ in tqdm(range(total_frames), desc="Reading frames", ncols=80):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        frame_landmarks = np.full((33, 3), np.nan)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[i] = [lm.x, lm.y, lm.z]
        
        all_landmarks.append(frame_landmarks)
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    if not all_landmarks:
        print("No pose landmarks detected in any frame.")
        return None

    all_landmarks = np.array(all_landmarks)
    interpolated_landmarks = interpolate_landmarks(all_landmarks)
    print("Landmarks have been succesfully interpolated !")

    # === For display : optionnal ===
    if show:
        for frame, landmarks in tqdm(zip(frames, interpolated_landmarks), total=len(frames), desc="Displaying", ncols=80):
            h, w, _ = frame.shape

            # Draw segments
            for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
                x1, y1, z1 = landmarks[start_idx]
                x2, y2, z2 = landmarks[end_idx]
                if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                    pt1 = (int(x1 * w), int(y1 * h))
                    pt2 = (int(x2 * w), int(y2 * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

            #Draw points
            for (x, y, z) in landmarks:
                if not np.isnan(x) and not np.isnan(y):
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            cv2.imshow("Pose Estimation (Interpolated)", frame)
            if cv2.waitKey(20) & 0xFF == 27:  # ESC to quit
                break

        cv2.destroyAllWindows()
        print("===== End of visualization =====")

    return interpolated_landmarks
    
    
    
    

def normalize_landmarks(landmarks):
    """
    Normalizes a landmarks dataframe so the coordinates don't depend on the camera angle.
    The origin of the coordinates is chosen as the hip center
    """

    
    T = len(landmarks)
    
    hip_center_orig = landmarks[:, [L_HP, R_HP], :].mean(axis=1)  # hip center will be the new origin
    landmarks_centered = landmarks - hip_center_orig[:, None, :] 
   
    shoulders_center = landmarks_centered[:, [L_SH, R_SH], :].mean(axis=1)   # shoulders center (in the centered coords)
    torso_len = np.linalg.norm(shoulders_center, axis=1)  # torso length: distance from hip(center=0) to shoulders_center
    torso_len[torso_len < 1e-6] = 1.0
    landmarks_centered = landmarks_centered / torso_len[:, None, None]   # scale to make torso length = 1 (or any stable unit)

    for t in range(T):
        pts = landmarks_centered[t]  # shape (33,3) in centered+scaled coords
        left_shoulder  = pts[L_SH]
        right_shoulder = pts[R_SH]
        left_hip       = pts[L_HP]
        right_hip      = pts[R_HP]

        # shoulder center in these coords (equivalently shoulders_center[t])
        shoulder_center = (left_shoulder + right_shoulder) * 0.5

        # x_axis: vector from left_shoulder -> right_shoulder (left->right)
        x_axis = right_shoulder - left_shoulder
        nx = np.linalg.norm(x_axis)
        if nx < 1e-6:
            # fallback: use global x unit
            x_axis = np.array([1.0, 0.0, 0.0])
            nx = 1.0
        x_axis = x_axis / nx
        # 1) X-axis : should always point from LEFT to RIGHT in camera space
        if x_axis[0] < 0:
            x_axis = -x_axis

        torso = shoulder_center - np.array([0.0, 0.0, 0.0])  # since hips are at origin after centering
        nt = np.linalg.norm(torso)
        if nt < 1e-6:
            torso = np.array([0.0, 1.0, 0.0])  # fallback
            nt = 1.0
        torso = torso / nt
        # z_axis: cross product to get perpendicular (x × torso)
        z_axis = np.cross(x_axis, torso)
        nz = np.linalg.norm(z_axis)
        if nz < 1e-6:
            # degenerate: choose arbitrary orthogonal vector
            if abs(x_axis[0]) < 0.9:
                z_axis = np.cross(x_axis, np.array([1.0, 0.0, 0.0]))
            else:
                z_axis = np.cross(x_axis, np.array([0.0, 1.0, 0.0]))
            nz = np.linalg.norm(z_axis)
            if nz < 1e-6:
                z_axis = np.array([0.0, 0.0, 1.0])
                nz = 1.0
        z_axis = z_axis / nz

        # Finally Y_axis: ensure right-handed system y = z × x
        y_axis = np.cross(z_axis, x_axis)
        ny = np.linalg.norm(y_axis)
        if ny < 1e-6:
            # fallback
            y_axis = np.array([0.0, 1.0, 0.0])
            ny = 1.0
        y_axis = y_axis / ny
        
        # 1) X-axis : should always point from LEFT to RIGHT in camera space
        # (doit aller de l'épaule gauche vers l'épaule droite)
        if x_axis[0] < 0:
            x_axis = -x_axis
            
        # 2) Y-axis : Should always point in the head direction
        head_vec = pts[HEAD] - (pts[L_HP] + pts[R_HP]) * 0.5
        if np.dot(head_vec, y_axis) < 0:
            y_axis = -y_axis

        # 3) Z-axis : We look at the wrist-shoulder axis so z is oriented in front of the person
        arms_vec = (pts[L_ELB] + pts[R_ELB]) * 0.5 - (pts[L_WR] + pts[R_WR]) * 0.5
        if np.dot(arms_vec, z_axis) > 0:
            z_axis = -z_axis

        # Assemble rotation matrix R where columns are basis vectors in camera coords:
        R = np.column_stack([x_axis, y_axis, z_axis])  
        # Project all points into this local frame: coords_local = R^T @ pts
        pts_local = (R.T @ pts.T).T   # shape (33,3)
        landmarks_centered[t] = pts_local
    return landmarks_centered




def angle_between(a, b, c):
    """
    Compute the angle ABC in degrees.
    a, b, c are 3D points (numpy arrays).
    """
    ba = a - b
    bc = c - b
    
    # Normalize vectors (avoid NaN)
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return np.nan
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1, 1)  
    return np.degrees(np.arccos(cosine_angle))





def compute_features_tensor(landmarks):
    """
    landmarks: numpy array (N_frames, 33, 3)
    returns: numpy array (N_frames, 17)
    """

    N = landmarks.shape[0]
    F = 19  # number of features
    features = np.zeros((N, F), dtype=np.float32)

    for t in range(N):
        pts = landmarks[t]

        torso = (pts[L_HP] + pts[R_HP]) / 2  # used for shoulder angle center

        # ----- Angles -----
        features[t, 0] = angle_between(pts[L_SH],  pts[L_ELB], pts[L_WR])                 # a_elb_L
        features[t, 1] = angle_between(pts[R_SH],  pts[R_ELB], pts[R_WR])                 # a_elb_R
        features[t, 2] = angle_between(pts[L_ELB], pts[L_SH],  torso)                     # a_sh_L
        features[t, 3] = angle_between(pts[R_ELB], pts[R_SH],  torso)                     # a_sh_R
        features[t, 4] = angle_between(pts[L_HP],  pts[L_KNEE], pts[L_ANK])               # a_kn_L
        features[t, 5] = angle_between(pts[R_HP],  pts[R_KNEE], pts[R_ANK])               # a_kn_R
        features[t, 6] = angle_between(pts[L_SH],  pts[L_HP],   pts[L_KNEE])              # a_hp_L
        features[t, 7] = angle_between(pts[R_SH],  pts[R_HP],   pts[R_KNEE])              # a_hp_R
        features[t, 8] = angle_between(pts[L_ELB], pts[L_WR],   pts[L_WR] + np.array([1,0,0])) # a_wr_L
        features[t, 9] = angle_between(pts[R_ELB], pts[R_WR],   pts[R_WR] + np.array([1,0,0])) # a_wr_R
        features[t, 10] = angle_between(pts[L_KNEE], pts[L_ANK], pts[L_FT])               # a_an_L
        features[t, 11] = angle_between(pts[R_KNEE], pts[R_ANK], pts[R_FT])               # a_an_R

        # ----- Distances -----
        features[t,12] = np.linalg.norm(pts[L_SH]   - pts[R_SH])                         # d_sh
        features[t,13] = np.linalg.norm(pts[L_HP]   - pts[R_HP])                         # d_hp
        features[t,14] = np.linalg.norm(pts[HEAD]   - ((pts[L_SH] + pts[R_SH]) / 2))     # d_head_sh
        features[t,15] = np.linalg.norm(pts[L_WR]   - pts[L_HP])                         # d_wr_hp
        features[t,16] = np.linalg.norm(pts[L_ANK]  - ((pts[L_HP] + pts[R_HP]) / 2))     # d_ank_hp
        features[t,17] = np.linalg.norm(pts[L_WR]   - pts[R_WR])                         # d_wr
        features[t,18] = np.linalg.norm(pts[L_KNEE] - pts[R_KNEE])                       # d_kn

    return features


def export_landmarks_and_features_csv(landmarks, features, video_path, label, output_csv_path):
    """
    landmarks: (N_frames, 33, 3)
    features: (N_frames, n_features)
    video_path: path to the video (for meta info)
    label: class label to insert
    output_csv_path: csv file to create
    """

    # ----- META INFO -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # ----- COLUMN NAMES -----
    video_name = os.path.basename(video_path)
    meta_cols = ['video_name','total_frames','frame_number','width','height','label']
    lm_cols = [f"lm_{i}_{axis}" 
               for i in range(33) 
               for axis in ['x', 'y', 'z']]
    feat_cols = [
        'a_elb_L','a_elb_R','a_sh_L','a_sh_R',
        'a_kn_L','a_kn_R','a_hp_L','a_hp_R',
        'a_wr_L','a_wr_R','a_an_L','a_an_R',
        'd_sh','d_hp','d_head_sh','d_wr_hp','d_ank_hp',
        'd_wr','d_kn'
    ]

    # Remove nonexistent cols if features has only 17 instead of 19
    feat_cols = feat_cols[:features.shape[1]]
    all_cols = meta_cols + lm_cols + feat_cols

    # ----- WRITE CSV -----
    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(all_cols)
        N = landmarks.shape[0]
        for frame_idx in range(N):
            # meta
            row = [
                video_name,
                total_frames,
                frame_idx,
                width,
                height,
                label
            ]
            # landmarks flatten: (33,3) → 99 values
            row.extend(landmarks[frame_idx].reshape(-1).tolist())
            # features
            row.extend(features[frame_idx].tolist())
            writer.writerow(row)

    print(f"CSV saved to {output_csv_path}")



def export_landmarks_and_features_df(landmarks, features, video_path, label=None):
    """
    landmarks: (N_frames, 33, 3)
    features:  (N_frames, n_features)
    video_path: path to video (meta info)
    label: class label (string or int)

    RETURN:
        df : pandas.DataFrame with all meta + landmarks + features
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    video_name = os.path.basename(video_path)
    N = landmarks.shape[0]

    meta_cols = ['video_name','total_frames','frame_number','width','height','label']
    lm_cols = [f"lm_{i}_{axis}" for i in range(33) for axis in ['x','y','z']]
    feat_cols = [
        'a_elb_L','a_elb_R','a_sh_L','a_sh_R',
        'a_kn_L','a_kn_R','a_hp_L','a_hp_R',
        'a_wr_L','a_wr_R','a_an_L','a_an_R',
        'd_sh','d_hp','d_head_sh','d_wr_hp','d_ank_hp',
        'd_wr','d_kn'
    ]

    feat_cols = feat_cols[:features.shape[1]]  # keep correct number

    # -------- BUILD DATAFRAME --------
    data = []

    label_value = label if label is not None else None
    
    for t in range(N):
        row = [
            video_name,
            total_frames,
            t,
            width,
            height,
            label_value
        ]
        row += landmarks[t].reshape(-1).tolist()  # flatten (33,3)
        row += features[t].tolist()               # n_features
        data.append(row)

    df = pd.DataFrame(data, columns = meta_cols + lm_cols + feat_cols)
    return df


def pipe_extract_landmark(filename): 
    landmarks = extract_pose_from_video_interpolated(filename, show = False)
    landmarks_norm = normalize_landmarks(landmarks)
    features = compute_features_tensor(landmarks_norm)
    df = export_landmarks_and_features_df(landmarks_norm, features, filename, None)
    return landmarks, df
    
    
if __name__ == "__main__":
    filename ="data/data-btc/push-up/push-up_ref.mp4"
    df = pipe_extract_landmark(filename)

    
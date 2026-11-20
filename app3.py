import glob
import os
import tempfile
import urllib.request

import cv2
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import streamlit as st


def configure_japanese_font():
    font_candidates = (
        glob.glob("/usr/share/fonts/**/ipag.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/IPAGothic.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/ipam.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True)
        + glob.glob("C:/Windows/Fonts/msgothic.ttc", recursive=False)
        + glob.glob("C:/Windows/Fonts/meiryo.ttc", recursive=False)
    )
    for font_path in font_candidates:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
        break
    plt.rcParams["axes.unicode_minus"] = False


configure_japanese_font()

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

IDX_NOSE = 0
IDX_L_HIP = 23
IDX_R_HIP = 24
IDX_L_KNEE = 25
IDX_R_KNEE = 26
IDX_L_ANKLE = 27
IDX_R_ANKLE = 28
IDX_L_FOOT = 31
IDX_R_FOOT = 32

POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


@st.cache_resource
def ensure_model_file(model_path: str = MODEL_PATH) -> str:
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def moving_average_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """欠損(NaN)を補間したうえで端落ち込みを抑えた奇数窓の移動平均を返す"""
    n = len(arr)
    if n == 0:
        return arr.copy()

    idx = np.arange(n)
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return np.full(n, np.nan, dtype=float)

    filled = np.interp(idx, idx[valid], arr[valid])
    win = max(1, int(window))
    if win % 2 == 0:
        win += 1
    if win == 1 or n < win:
        return filled

    pad = win // 2
    padded = np.pad(filled, pad_width=pad, mode="edge")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(padded, kernel, mode="valid")


def calc_knee_inward_from_hip_ankle_line(hip, knee, ankle, side: str) -> float:
    """
    股関節(hip)と足首(ankle)を結ぶ直線上の、膝の高さ(knee[1])におけるx座標を求め、
    基準線より内側に入っている度合い [px]（正＝内側、負＝外側）を返す。
    side: "left"（被験者左脚＝画面右寄り） or "right"（被験者右脚＝画面左寄り）
    """
    if any(np.isnan(pt).any() for pt in (hip, knee, ankle)):
        return np.nan

    dy = ankle[1] - hip[1]
    if abs(dy) < 1e-6:
        x_line = ankle[0]
    else:
        t = (knee[1] - hip[1]) / dy
        x_line = hip[0] + t * (ankle[0] - hip[0])

    if side == "right":
        return float(knee[0] - x_line)
    else:
        return float(x_line - knee[0])


def draw_pose_skeleton(img_bgr: np.ndarray, pts_px: np.ndarray):
    for i, j in POSE_EDGES:
        p1, p2 = pts_px[i], pts_px[j]
        if not (np.isnan(p1).any() or np.isnan(p2).any()):
            cv2.line(img_bgr, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 200, 0), 2)
    for pt in pts_px:
        if not np.isnan(pt).any():
            cv2.circle(img_bgr, (int(pt[0]), int(pt[1])), 4, (0, 140, 255), -1)


def estimate_standing_height_px(
    nose_y_smooth: np.ndarray,
    l_foot_y_smooth: np.ndarray,
    r_foot_y_smooth: np.ndarray,
    standing_frames: int = 10,
) -> float:
    """初期立位区間におけるNOSE yと足部yの差から仮の鉛直身長 [px] を算出する"""
    n = len(nose_y_smooth)
    end_idx = max(1, min(n, int(standing_frames)))

    nose_stand = float(np.nanmedian(nose_y_smooth[:end_idx]))
    foot_combined = np.nanmax(np.vstack([l_foot_y_smooth[:end_idx], r_foot_y_smooth[:end_idx]]), axis=0)
    foot_stand = float(np.nanmedian(foot_combined))

    height_px = foot_stand - nose_stand
    if np.isnan(height_px) or height_px <= 10.0:
        raise ValueError("初期立位区間の鉛直身長(px)を算出できませんでした。")
    return float(height_px)


def analyze_lunge_with_knee_line(
    video_path: str,
    height_cm: float = 170.0,
    standing_frames: int = 10,
    ma_window: int = 5,
    min_det_conf: float = 0.5,
    min_trk_conf: float = 0.5,
):
    model_file = ensure_model_file(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    frames_bgr = []
    all_pts_px = []

    base_options = mp_python.BaseOptions(model_asset_path=model_file)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_det_conf,
        min_tracking_confidence=min_trk_conf,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)

            pts_px = np.full((33, 2), np.nan, dtype=float)
            rec = {
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "nose_y": np.nan,
                "hip_mid_y": np.nan,
                "l_hip": np.full(2, np.nan),
                "l_knee": np.full(2, np.nan),
                "l_ankle": np.full(2, np.nan),
                "l_foot": np.full(2, np.nan),
                "r_hip": np.full(2, np.nan),
                "r_knee": np.full(2, np.nan),
                "r_ankle": np.full(2, np.nan),
                "r_foot": np.full(2, np.nan),
                "l_inward_px": np.nan,
                "r_inward_px": np.nan,
            }

            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                for i, lm in enumerate(res.pose_landmarks[0]):
                    pts_px[i] = [lm.x * w, lm.y * h]

                l_hip, l_knee, l_ankle, l_foot = pts_px[IDX_L_HIP], pts_px[IDX_L_KNEE], pts_px[IDX_L_ANKLE], pts_px[IDX_L_FOOT]
                r_hip, r_knee, r_ankle, r_foot = pts_px[IDX_R_HIP], pts_px[IDX_R_KNEE], pts_px[IDX_R_ANKLE], pts_px[IDX_R_FOOT]

                rec["nose_y"] = pts_px[IDX_NOSE, 1]
                rec["hip_mid_y"] = 0.5 * (l_hip[1] + r_hip[1])
                rec["l_hip"], rec["l_knee"], rec["l_ankle"], rec["l_foot"] = l_hip, l_knee, l_ankle, l_foot
                rec["r_hip"], rec["r_knee"], rec["r_ankle"], rec["r_foot"] = r_hip, r_knee, r_ankle, r_foot

                rec["l_inward_px"] = calc_knee_inward_from_hip_ankle_line(l_hip, l_knee, l_ankle, "left")
                rec["r_inward_px"] = calc_knee_inward_from_hip_ankle_line(r_hip, r_knee, r_ankle, "right")

            all_pts_px.append(pts_px)
            rows.append(rec)
            frame_idx += 1

    cap.release()
    if len(rows) == 0:
        raise ValueError("動画フレームを読み込めませんでした。")

    times = np.array([r["time_s"] for r in rows], dtype=float)
    nose_y_smooth = moving_average_1d(np.array([r["nose_y"] for r in rows], dtype=float), ma_window)
    hip_y_smooth = moving_average_1d(np.array([r["hip_mid_y"] for r in rows], dtype=float), ma_window)

    l_inward_smooth_px = moving_average_1d(np.array([r["l_inward_px"] for r in rows], dtype=float), ma_window)
    r_inward_smooth_px = moving_average_1d(np.array([r["r_inward_px"] for r in rows], dtype=float), ma_window)

    l_foot_y_smooth = moving_average_1d(
        np.array([np.nanmax([r["l_foot"][1], r["l_ankle"][1]]) for r in rows], dtype=float),
        ma_window,
    )
    r_foot_y_smooth = moving_average_1d(
        np.array([np.nanmax([r["r_foot"][1], r["r_ankle"][1]]) for r in rows], dtype=float),
        ma_window,
    )

    standing_height_px = estimate_standing_height_px(
        nose_y_smooth, l_foot_y_smooth, r_foot_y_smooth, standing_frames=standing_frames
    )
    cm_per_px = float(height_cm) / standing_height_px

    bottom_idx = int(np.nanargmax(nose_y_smooth))
    bottom_rec = rows[bottom_idx]

    l_front_score = np.nanmax([bottom_rec["l_foot"][1], bottom_rec["l_ankle"][1]])
    r_front_score = np.nanmax([bottom_rec["r_foot"][1], bottom_rec["r_ankle"][1]])
    lead_side = "left" if (np.isnan(r_front_score) or l_front_score >= r_front_score) else "right"

    lead_inward_smooth_px = l_inward_smooth_px if lead_side == "left" else r_inward_smooth_px
    l_inward_smooth_cm = l_inward_smooth_px * cm_per_px
    r_inward_smooth_cm = r_inward_smooth_px * cm_per_px
    lead_inward_smooth_cm = lead_inward_smooth_px * cm_per_px

    crossing_indices = []
    for i in range(1, len(lead_inward_smooth_cm)):
        prev_v, curr_v = lead_inward_smooth_cm[i - 1], lead_inward_smooth_cm[i]
        if not (np.isnan(prev_v) or np.isnan(curr_v)):
            if prev_v <= 0.0 and curr_v > 0.0:
                crossing_indices.append(i)

    dt = 1.0 / fps
    hip_upward_vel_px_s = -np.gradient(hip_y_smooth, dt)
    hip_upward_vel_cm_s = hip_upward_vel_px_s * cm_per_px

    nose_stand_y = float(np.nanmedian(nose_y_smooth[: max(1, min(len(nose_y_smooth), standing_frames))]))
    nose_disp_cm = (nose_y_smooth - nose_stand_y) * cm_per_px

    bottom_vis = frames_bgr[bottom_idx].copy()
    draw_pose_skeleton(bottom_vis, all_pts_px[bottom_idx])

    hip_pt = bottom_rec["l_hip"] if lead_side == "left" else bottom_rec["r_hip"]
    knee_pt = bottom_rec["l_knee"] if lead_side == "left" else bottom_rec["r_knee"]
    ankle_pt = bottom_rec["l_ankle"] if lead_side == "left" else bottom_rec["r_ankle"]

    if not (np.isnan(hip_pt).any() or np.isnan(ankle_pt).any()):
        cv2.line(
            bottom_vis,
            (int(hip_pt[0]), int(hip_pt[1])),
            (int(ankle_pt[0]), int(ankle_pt[1])),
            (0, 255, 255),
            2,
        )
    if not

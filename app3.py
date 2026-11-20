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
    if not np.isnan(knee_pt).any():
        cv2.circle(bottom_vis, (int(knee_pt[0]), int(knee_pt[1])), 9, (0, 0, 255), 3)

    return {
        "fps": fps,
        "total_frames": len(rows),
        "times": times,
        "height_cm": float(height_cm),
        "standing_height_px": standing_height_px,
        "cm_per_px": cm_per_px,
        "bottom_idx": bottom_idx,
        "bottom_time_s": times[bottom_idx],
        "lead_side": lead_side,
        "l_inward_smooth_cm": l_inward_smooth_cm,
        "r_inward_smooth_cm": r_inward_smooth_cm,
        "lead_inward_smooth_cm": lead_inward_smooth_cm,
        "l_foot_y_smooth": l_foot_y_smooth,
        "r_foot_y_smooth": r_foot_y_smooth,
        "crossing_indices": crossing_indices,
        "hip_upward_vel_cm_s": hip_upward_vel_cm_s,
        "nose_disp_cm": nose_disp_cm,
        "bottom_frame_bgr": bottom_vis,
    }


def detect_touchdown_and_ascent_onset(out: dict):
    b_idx = int(out["bottom_idx"])
    vel = np.asarray(out["hip_upward_vel_cm_s"], dtype=float)

    lead_foot_y = (
        np.asarray(out["l_foot_y_smooth"], dtype=float)
        if out["lead_side"] == "left"
        else np.asarray(out["r_foot_y_smooth"], dtype=float)
    )
    search_end = max(1, b_idx + 1)
    if np.any(~np.isnan(lead_foot_y[:search_end])):
        td_idx = int(np.nanargmax(lead_foot_y[:search_end]))
    else:
        td_idx = max(0, b_idx - 10)

    pre_margin = max(0, b_idx - 3)
    peak_slice = vel[b_idx:] if b_idx < len(vel) else vel[pre_margin:]
    peak_vel = float(np.nanmax(peak_slice)) if len(peak_slice) > 0 else 0.0

    onset_idx = b_idx
    if peak_vel > 0:
        thresh = 0.15 * peak_vel
        after_thresh = np.where((np.arange(len(vel)) >= pre_margin) & (vel >= thresh))[0]
        if len(after_thresh) > 0:
            first_rise = int(after_thresh[0])
            back_candidates = np.where((np.arange(len(vel)) <= first_rise) & (vel <= 0.0))[0]
            if len(back_candidates) > 0:
                onset_idx = int(back_candidates[-1])
            else:
                onset_idx = max(pre_margin, first_rise - 1)

    return td_idx, onset_idx


def build_summary_figure(out: dict, td_idx: int, onset_idx: int, ma_window: int = 5):
    t = out["times"]
    b_idx = int(out["bottom_idx"])
    lead_jp = "左脚" if out["lead_side"] == "left" else "右脚"

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.8), sharex=True)

    event_lines = [
        (t[td_idx], "#8c564b", "-.", "接地"),
        (t[b_idx], "#2ca02c", "--", "最下点"),
        (t[onset_idx], "#9467bd", "--", "上昇開始"),
    ]

    def add_event_vlines(ax, include_legend=False):
        for x_val, col, ls, lbl in event_lines:
            ax.axvline(
                x_val,
                color=col,
                linestyle=ls,
                linewidth=1.3,
                alpha=0.85,
                label=lbl if include_legend else "_nolegend_",
            )

    lead_vals = np.asarray(out["lead_inward_smooth_cm"], dtype=float)
    other_vals = np.asarray(
        out["r_inward_smooth_cm"] if out["lead_side"] == "left" else out["l_inward_smooth_cm"],
        dtype=float,
    )

    axes[0].plot(
        t,
        lead_vals,
        color="#d62728",
        linewidth=2.0,
        label=f"前脚（{lead_jp}）",
    )
    axes[0].plot(
        t,
        other_vals,
        color="#999999",
        linewidth=1.2,
        linestyle=":",
        label="反対脚",
    )
    axes[0].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, label="基準線 (0 cm)")
    for idx in out.get("crossing_indices", []):
        axes[0].scatter(
            [t[idx]],
            [lead_vals[idx]],
            color="purple",
            zorder=5,
            label="＋突入" if idx == out["crossing_indices"][0] else "_nolegend_",
        )

    all_inward = np.concatenate([lead_vals, other_vals])
    valid_inward = all_inward[~np.isnan(all_inward)]
    if len(valid_inward) > 0:
        y_min = min(float(np.min(valid_inward)) - 0.8, -1.0)
        y_max = max(float(np.max(valid_inward)) + 0.8, 1.0)
    else:
        y_min, y_max = -2.0, 2.0
    axes[0].set_ylim(y_min, y_max)
    if y_max > 0.0:
        axes[0].axhspan(0.0, y_max, color="#ff4d4f", alpha=0.22, zorder=0, label="＋域（ニーイン注意）")

    add_event_vlines(axes[0], include_legend=True)
    axes[0].set_title("ニーインの度合い（+：ニーイン，-：ニーアウト）")
    axes[0].set_ylabel("度合い (cm)")
    axes[0].grid(True, alpha=0.30, zorder=1)
    axes[0].legend(loc="best", fontsize=8.0)

    axes[1].plot(
        t[b_idx:],
        out["hip_upward_vel_cm_s"][b_idx:],
        color="#1f77b4",
        linewidth=2.0,
        label="上昇速度",
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    add_event_vlines(axes[1], include_legend=True)
    axes[1].set_title("立ち上がり上昇速度")
    axes[1].set_ylabel("速度 (cm/s)")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="best", fontsize=8.0)

    axes[2].plot(
        t,
        out["nose_disp_cm"],
        color="#ff7f0e",
        linewidth=2.0,
        label=f"鉛直移動（移動平均{ma_window}）",
    )
    add_event_vlines(axes[2], include_legend=True)
    axes[2].invert_yaxis()
    axes[2].set_title("鉛直移動（下向きが＋表示）")
    axes[2].set_xlabel("時間 (s)")
    axes[2].set_ylabel("移動量 (cm)")
    axes[2].grid(True, alpha=0.35)
    axes[2].legend(loc="best", fontsize=8.0)

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="フォワードランジ解析", layout="wide")
    st.title("フォワードランジ解析")
    st.caption("ニーイン・上昇速度・鉛直移動の確認")

    with st.sidebar:
        st.header("設定")
        uploaded_file = st.file_uploader(
            "動画をアップロード",
            type=["mp4", "mov", "avi", "m4v"],
        )
        height_cm = st.number_input(
            "身長 (cm)",
            min_value=100.0,
            max_value=230.0,
            value=170.0,
            step=0.5,
        )
        standing_frames = 10
        ma_window = 5
        run_clicked = st.button("解析を実行", type="primary", use_container_width=True)

    st.markdown(
        """
        <div style="padding: 16px 20px; border-left: 6px solid #d97706; background-color: #fffbeb; border-radius: 6px; margin-bottom: 14px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #92400e; margin-bottom: 8px;">
                動画撮影のお願い（正確な解析のために）
            </div>
            <ul style="font-size: 1.05rem; line-height: 1.8; color: #1f2937; margin: 0; padding-left: 1.4rem;">
                <li><strong>縦置き</strong>で撮影してください</li>
                <li><strong>カメラと正対</strong>して撮影してください</li>
                <li><strong>カメラの水平</strong>をしっかりとってください</li>
                <li><strong>体全体（頭〜足先）が見切れない</strong>ように映してください</li>
            </ul>
        </div>

        <div style="padding: 16px 20px; border-left: 6px solid #2563eb; background-color: #eff6ff; border-radius: 6px; margin-bottom: 18px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #1e40af; margin-bottom: 8px;">
                フォワードランジの実施方法
            </div>
            <p style="font-size: 1.05rem; line-height: 1.8; color: #1f2937; margin: 0;">
                足を肩幅に自然と開き、一歩前に踏み込みます。最下点では前と後ろの膝がどちらも90度近くなるようにします。膝を地面につかずに、しっかりと地面を蹴って立ち上がり、元の姿勢に戻ります。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        st.info("左から動画をアップロードしてください。")
        return

    if not run_clicked and "lunge_out" not in st.session_state:
        st.info("「解析を実行」を押してください。")
        return

    if run_clicked:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("解析中..."):
                out = analyze_lunge_with_knee_line(
                    video_path=tmp_path,
                    height_cm=height_cm,
                    standing_frames=standing_frames,
                    ma_window=ma_window,
                )
            st.session_state["lunge_out"] = out
            st.session_state["ma_window"] = ma_window
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if "lunge_out" not in st.session_state:
        return

    out = st.session_state["lunge_out"]
    win = st.session_state.get("ma_window", ma_window)
    t = out["times"]
    b_idx = int(out["bottom_idx"])
    td_idx, onset_idx = detect_touchdown_and_ascent_onset(out)
    lead_jp = "左脚" if out["lead_side"] == "left" else "右脚"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("前脚", lead_jp)
    c2.metric("接地", f"{t[td_idx]:.2f} s ({td_idx} f)")
    c3.metric("最下点", f"{t[b_idx]:.2f} s ({b_idx} f)")
    c4.metric("上昇開始", f"{t[onset_idx]:.2f} s ({onset_idx} f)")

    col_graph, col_img = st.columns([1.35, 1.0])

    with col_graph:
        st.subheader("波形グラフ")
        fig = build_summary_figure(out, td_idx=td_idx, onset_idx=onset_idx, ma_window=win)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_img:
        st.subheader("最下点の骨格・基準線")
        bottom_rgb = cv2.cvtColor(out["bottom_frame_bgr"], cv2.COLOR_BGR2RGB)
        st.image(bottom_rgb, caption=f"最下点：{b_idx} f / {t[b_idx]:.2f} s", use_container_width=True)


if __name__ == "__main__":
    main()

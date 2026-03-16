"""
beard_ar.py  —  Snapchat-style Beard AR Overlay (CLI mode)
MediaPipe >= 0.10  (new Tasks API — no mp.solutions)
============================================================

FIRST RUN — model file download (one time, ~30MB):
  Script auto-downloads face_landmarker.task into same folder.
  Or manually:
    wget 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -O face_landmarker.task

Controls:
  1-5  : Switch beard style
  [    : Shrink beard
  ]    : Grow beard
  d    : Toggle debug landmarks
  q    : Quit

Beard Styles:
  1 - Full Beard
  2 - Goatee
  3 - Stubble
  4 - Viking / Long Beard
  5 - Handlebar Moustache
"""

import cv2
import numpy as np
import os
import sys
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# ──────────────────────────────────────────────
# MODEL AUTO-DOWNLOAD
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("[INFO] Downloading face_landmarker model (~30 MB) ...")
    print(f"       Saving to: {MODEL_PATH}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Download complete.")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("Manual download:")
        print(f"  wget '{MODEL_URL}' -O face_landmarker.task")
        sys.exit(1)

# ──────────────────────────────────────────────
# BEARD COLORS  (BGR)
# ──────────────────────────────────────────────
C = {
    "dark":      (20,  35,  55),
    "mid":       (40,  65,  95),
    "highlight": (70, 100, 140),
    "stubble":   (50,  60,  70),
    "mustache":  (15,  28,  45),
}

# ──────────────────────────────────────────────
# KEY LANDMARK INDICES  (MediaPipe 478-point mesh)
# ──────────────────────────────────────────────
IDX = {
    "chin_tip":      152,
    "chin_left":     397,
    "chin_right":    172,
    "jaw_left":      132,
    "jaw_right":     361,
    "mouth_left":    61,
    "mouth_right":   291,
    "upper_lip_top": 0,
    "lower_lip_bot": 17,
    "nose_bottom":   2,
}

# ──────────────────────────────────────────────
# GEOMETRY HELPERS
# ──────────────────────────────────────────────
def lerp(p1, p2, t):
    return (int(p1[0] + (p2[0]-p1[0])*t),
            int(p1[1] + (p2[1]-p1[1])*t))

def down(pt, d):
    return (pt[0], pt[1] + int(d))

def mid(p1, p2):
    return ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)

def resolve_anchors(face_lm_list, w, h):
    """
    face_lm_list: list of NormalizedLandmark from FaceLandmarkerResult
    Returns dict of anchor_name -> (pixel_x, pixel_y)
    """
    out = {}
    for name, idx in IDX.items():
        lm = face_lm_list[idx]
        out[name] = (int(lm.x * w), int(lm.y * h))
    return out

# ──────────────────────────────────────────────
# BEARD DRAW FUNCTIONS
# Each: (frame, anchors_dict, scale_float) -> None
# ──────────────────────────────────────────────

def draw_full_beard(frame, a, scale):
    chin    = a["chin_tip"]
    jaw_l   = a["jaw_left"]
    jaw_r   = a["jaw_right"]
    mouth_l = a["mouth_left"]
    mouth_r = a["mouth_right"]
    ul      = a["upper_lip_top"]
    chin_l  = a["chin_left"]
    chin_r  = a["chin_right"]
    drop    = int(35 * scale)

    pts = np.array([
        jaw_l,
        lerp(jaw_l, chin, 0.3),
        lerp(chin_l, chin, 0.5),
        down(chin, drop),
        lerp(chin_r, chin, 0.5),
        lerp(jaw_r, chin, 0.3),
        jaw_r,
        mouth_r,
        ul,
        mouth_l,
    ], dtype=np.int32)

    ov = frame.copy()
    cv2.fillPoly(ov, [pts], C["dark"])
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)

    # Mustache
    mu = np.array([
        mouth_l, lerp(mouth_l, ul, 0.5), ul,
        lerp(mouth_r, ul, 0.5), mouth_r,
        lerp(mouth_l, mouth_r, 0.5),
    ], dtype=np.int32)
    cv2.fillPoly(frame, [mu], C["mustache"])

    # Hair texture strokes
    for i in range(8):
        t = i / 8
        for base in [jaw_l, jaw_r]:
            p1 = lerp(base, chin, t)
            p2 = lerp(p1, down(p1, drop//2), 0.4)
            cv2.line(frame, p1, p2, C["highlight"], 1, cv2.LINE_AA)
    cv2.line(frame, chin, down(chin, drop), C["highlight"], 2, cv2.LINE_AA)


def draw_goatee(frame, a, scale):
    chin    = a["chin_tip"]
    mouth_l = a["mouth_left"]
    mouth_r = a["mouth_right"]
    ul      = a["upper_lip_top"]
    chin_l  = a["chin_left"]
    chin_r  = a["chin_right"]
    drop    = int(30 * scale)

    le = lerp(mouth_l, chin_l, 0.35)
    re = lerp(mouth_r, chin_r, 0.35)

    pts = np.array([
        le, mouth_l, ul, mouth_r, re,
        down(lerp(chin, chin_r, 0.15), drop),
        down(chin, drop + int(8*scale)),
        down(lerp(chin, chin_l, 0.15), drop),
    ], dtype=np.int32)

    ov = frame.copy()
    cv2.fillPoly(ov, [pts], C["dark"])
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, chin, down(chin, drop+int(8*scale)), C["highlight"], 2, cv2.LINE_AA)

    # Twin ellipse mustache
    mu_y  = ul[1] - int(2 * scale)
    mw    = int((mouth_r[0] - mouth_l[0]) * 0.48)
    thick = max(2, int(4 * scale))
    for cx_offset in [mouth_l[0] + mw//2, mouth_r[0] - mw//2]:
        cv2.ellipse(frame, (cx_offset, mu_y), (mw, max(1,int(5*scale))),
                    0, 200, 340, C["mustache"], thick, cv2.LINE_AA)


def draw_stubble(frame, a, scale):
    chin    = a["chin_tip"]
    jaw_l   = a["jaw_left"]
    jaw_r   = a["jaw_right"]
    mouth_l = a["mouth_left"]
    mouth_r = a["mouth_right"]
    ul      = a["upper_lip_top"]

    mask_pts = np.array([
        jaw_l, mouth_l, ul, mouth_r, jaw_r,
        down(chin, int(5*scale)),
    ], dtype=np.int32)

    stub = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(stub, [mask_pts], 255)

    ys, xs = np.where(stub > 0)
    if len(xs) > 0:
        n = int(len(xs) * 0.12)
        chosen = np.random.choice(len(xs), size=min(n, len(xs)), replace=False)
        for idx in chosen:
            px, py = int(xs[idx]), int(ys[idx])
            r  = np.random.randint(1, max(2, int(2*scale)))
            al = np.random.uniform(0.4, 0.9)
            col = tuple(int(c * al) for c in C["stubble"])
            cv2.circle(frame, (px, py), r, col, -1, cv2.LINE_AA)

    # Thin mustache shadow
    mu_pts = np.array([
        (mouth_l[0], ul[1]),
        (mid(mouth_l, mouth_r)[0], ul[1] - int(3*scale)),
        (mouth_r[0], ul[1]),
    ], dtype=np.int32)
    cv2.polylines(frame, [mu_pts], False, C["stubble"],
                  max(2, int(5*scale)), cv2.LINE_AA)


def draw_viking(frame, a, scale):
    chin    = a["chin_tip"]
    jaw_l   = a["jaw_left"]
    jaw_r   = a["jaw_right"]
    mouth_l = a["mouth_left"]
    mouth_r = a["mouth_right"]
    ul      = a["upper_lip_top"]
    chin_l  = a["chin_left"]
    chin_r  = a["chin_right"]
    drop    = int(80 * scale)
    gap     = int(18 * scale)
    bw      = int(16 * scale)

    base = np.array([
        jaw_l, mouth_l, ul, mouth_r, jaw_r,
        lerp(jaw_r, chin, 0.6), chin_r,
        down(chin, int(25*scale)),
        chin_l, lerp(jaw_l, chin, 0.6),
    ], dtype=np.int32)

    ov = frame.copy()
    cv2.fillPoly(ov, [base], C["dark"])
    cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)

    # Twin braids
    for bx in [chin[0]-gap, chin[0]+gap]:
        braid = np.array([
            (bx-bw, chin[1]+int(10*scale)),
            (bx+bw, chin[1]+int(10*scale)),
            (bx+bw//2, chin[1]+drop),
            (bx-bw//2, chin[1]+drop),
        ], dtype=np.int32)
        cv2.fillPoly(frame, [braid], C["mid"])
        for i in range(4):
            y = chin[1] + int(scale*(15+i*18))
            cv2.line(frame, (bx-bw,y), (bx+bw,y), C["highlight"], 1, cv2.LINE_AA)
        cv2.circle(frame, (bx, chin[1]+drop), int(7*scale), C["mid"], -1, cv2.LINE_AA)

    # Thick mustache
    mu = np.array([
        mouth_l,
        (mouth_l[0]-int(8*scale), ul[1]-int(2*scale)),
        mid(mouth_l, ul), ul, mid(mouth_r, ul),
        (mouth_r[0]+int(8*scale), ul[1]-int(2*scale)),
        mouth_r,
    ], dtype=np.int32)
    cv2.polylines(frame, [mu], False, C["mustache"], int(6*scale), cv2.LINE_AA)


def draw_handlebar(frame, a, scale):
    ul      = a["upper_lip_top"]
    mouth_l = a["mouth_left"]
    mouth_r = a["mouth_right"]
    mu_y    = ul[1]
    mu_w    = int((mouth_r[0] - mouth_l[0]) * 0.55)
    thick   = max(4, int(9 * scale))
    curl    = int(18 * scale)
    cx      = (mouth_l[0] + mouth_r[0]) // 2

    cv2.line(frame, (cx-mu_w,mu_y), (cx+mu_w,mu_y), C["mustache"], thick, cv2.LINE_AA)

    def bezier(p0, p1, p2, p3, steps=20):
        pts = []
        for i in range(steps+1):
            t = i/steps
            x = int((1-t)**3*p0[0]+3*(1-t)**2*t*p1[0]+3*(1-t)*t**2*p2[0]+t**3*p3[0])
            y = int((1-t)**3*p0[1]+3*(1-t)**2*t*p1[1]+3*(1-t)*t**2*p2[1]+t**3*p3[1])
            pts.append((x,y))
        return np.array(pts, dtype=np.int32)

    lc = bezier((cx-mu_w,mu_y),
                (cx-mu_w-int(8*scale), mu_y-int(4*scale)),
                (cx-mu_w-curl, mu_y-int(16*scale)),
                (cx-mu_w-curl+int(5*scale), mu_y-int(20*scale)))
    rc = bezier((cx+mu_w,mu_y),
                (cx+mu_w+int(8*scale), mu_y-int(4*scale)),
                (cx+mu_w+curl, mu_y-int(16*scale)),
                (cx+mu_w+curl-int(5*scale), mu_y-int(20*scale)))

    for curve in [lc, rc]:
        cv2.polylines(frame, [curve], False, C["mustache"],
                      max(3,int(6*scale)), cv2.LINE_AA)

    arch = np.array([
        mouth_l,
        (cx-int(mu_w*0.5), mu_y-int(4*scale)),
        (cx, mu_y-int(7*scale)),
        (cx+int(mu_w*0.5), mu_y-int(4*scale)),
        mouth_r,
    ], dtype=np.int32)
    cv2.polylines(frame, [arch], False, C["mustache"],
                  max(2,int(5*scale)), cv2.LINE_AA)


# ──────────────────────────────────────────────
# STYLE REGISTRY
# ──────────────────────────────────────────────
STYLES = {
    1: ("Full Beard",          draw_full_beard),
    2: ("Goatee",              draw_goatee),
    3: ("Stubble",             draw_stubble),
    4: ("Viking / Long Beard", draw_viking),
    5: ("Handlebar Moustache", draw_handlebar),
}

# ──────────────────────────────────────────────
# HUD + DEBUG
# ──────────────────────────────────────────────
def draw_hud(frame, style_id, scale):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h-44), (w, h), (15,15,15), -1)
    txt = (f"Style {style_id}/5: {STYLES[style_id][0]}   "
           f"Scale:{scale:.1f}   "
           f"[1-5]style  [/]scale  [d]debug  [q]quit")
    cv2.putText(frame, txt, (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (200,200,200), 1, cv2.LINE_AA)

def draw_debug(frame, face_lm_list, w, h):
    for name, idx in IDX.items():
        lm = face_lm_list[idx]
        px = (int(lm.x*w), int(lm.y*h))
        cv2.circle(frame, px, 3, (0,255,0), -1, cv2.LINE_AA)
        cv2.putText(frame, name[:6], (px[0]+4, px[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,255,0), 1)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    download_model()

    opts = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    WIN = "Beard AR  |  Keys: 1-5=style  [/]=scale  d=debug  q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    style  = 1
    scale  = 1.0
    debug  = False
    ts_ms  = 0

    print("=" * 52)
    print("  Beard AR  —  MediaPipe 0.10+ (Tasks API)")
    print("=" * 52)
    for k, (n, _) in STYLES.items():
        print(f"  [{k}]  {n}")
    print("  [ / ]  scale   [d] debug   [q] quit")
    print("=" * 52)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        ts_ms += 33
        result = landmarker.detect_for_video(mp_img, ts_ms)

        if result.face_landmarks:
            face_lm = result.face_landmarks[0]   # list of NormalizedLandmark
            anchors = resolve_anchors(face_lm, w, h)
            STYLES[style][1](frame, anchors, scale)
            if debug:
                draw_debug(frame, face_lm, w, h)
        else:
            cv2.putText(frame, "No face detected",
                        (w//2 - 100, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,220), 2)

        draw_hud(frame, style, scale)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'):
            break
        elif key in [ord(str(i)) for i in range(1, 6)]:
            style = int(chr(key))
            print(f"  -> {STYLES[style][0]}")
        elif key == ord(']'):
            scale = min(round(scale + 0.1, 1), 2.5)
            print(f"  Scale: {scale}")
        elif key == ord('['):
            scale = max(round(scale - 0.1, 1), 0.4)
            print(f"  Scale: {scale}")
        elif key == ord('d'):
            debug = not debug
            print(f"  Debug: {'ON' if debug else 'OFF'}")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

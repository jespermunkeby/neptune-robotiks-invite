"""
Generate mosh maps for datamosh displacement.

For each image, produces an RGB texture packed as:
  R = horizontal displacement (128=zero, 0=hard left, 255=hard right)
  G = vertical displacement (128=zero, 0=hard up, 255=hard down)
  B = edge distance field (0=on edge, 255=far from edge)

Displacement comes from:
  1. Optical flow between consecutive images (real motion vectors)
  2. Image gradient direction (Sobel) as fallback/mix — pushes pixels
     along the steepest intensity change direction
"""
import os, glob
import cv2
import numpy as np

IMG_DIR = "images"
MOSH_DIR = "mosh"
os.makedirs(MOSH_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(IMG_DIR, "*.webp")))
N = len(files)
print(f"Found {N} images")

WORK_W, WORK_H = 480, 480

def load(path):
    img = cv2.imread(path)
    if img is None:
        return None, None
    small = cv2.resize(img, (WORK_W, WORK_H))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return img, gray

def compute_flow(gray_a, gray_b):
    return cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=5, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

def compute_gradient_flow(gray):
    """Sobel gradient direction — pixels want to move along intensity gradients."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.sqrt(gx**2 + gy**2) + 1e-8
    gx = (gx / mag) * np.clip(mag / mag.max(), 0, 1) * 15
    gy = (gy / mag) * np.clip(mag / mag.max(), 0, 1) * 15
    return np.stack([gx, gy], axis=-1)

def compute_edge_distance(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 25, 80)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    inv = cv2.bitwise_not(edges)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
    dist = dist / (dist.max() + 1e-8)
    dist = cv2.GaussianBlur(dist, (9, 9), 0)
    return (dist * 255).astype(np.uint8)

prev_gray = None

for i, fpath in enumerate(files):
    fname = os.path.basename(fpath)
    out_path = os.path.join(MOSH_DIR, fname)

    orig, gray = load(fpath)
    if gray is None:
        print(f"  [{i+1}/{N}] {fname} — failed")
        continue

    # Optical flow from prev → this (at fixed 480x480 so shapes always match)
    if prev_gray is not None:
        oflow = compute_flow(prev_gray, gray)
    else:
        oflow = np.zeros((WORK_H, WORK_W, 2), dtype=np.float32)

    # Gradient-based displacement
    gflow = compute_gradient_flow(gray)

    # Blend: optical flow + gradient flow
    flow_mag = np.sqrt(oflow[:,:,0]**2 + oflow[:,:,1]**2)
    avg_flow = flow_mag.mean()

    # If optical flow is weak, lean more on gradient flow
    flow_weight = np.clip(avg_flow / 3.0, 0.0, 0.7)
    flow = oflow * (0.4 + flow_weight * 0.6) + gflow * (0.6 - flow_weight * 0.6)

    # Encode: clamp to [-20,20], map to [0,255]
    fx = np.clip(flow[:,:,0], -20, 20)
    fy = np.clip(flow[:,:,1], -20, 20)
    r = ((fx / 20.0) * 127 + 128).astype(np.uint8)
    g = ((fy / 20.0) * 127 + 128).astype(np.uint8)
    b = compute_edge_distance(gray)

    # Pack BGR (OpenCV order) and resize to original image dimensions
    mosh = cv2.merge([b, g, r])
    oh, ow = orig.shape[:2]
    mosh_full = cv2.resize(mosh, (ow, oh), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(out_path, mosh_full, [cv2.IMWRITE_WEBP_QUALITY, 90])

    fx_r = fx.max() - fx.min()
    fy_r = fy.max() - fy.min()
    print(f"  [{i+1}/{N}] {fname} — flow_avg={avg_flow:.1f} grad_blend={1-flow_weight:.0%} range=({fx_r:.0f},{fy_r:.0f})")

    prev_gray = gray

print(f"\nDone → {MOSH_DIR}/")

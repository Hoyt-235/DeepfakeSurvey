import os
import cv2
import dlib
import argparse
import random
import uuid
import csv
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== WORKER INITIALIZER =====
def worker_init():
    """Initialize face detector for worker."""
    global _detector
    cv2.setNumThreads(1)
    _detector = dlib.get_frontal_face_detector()

# ===== FACE EXTRACTION =====
def extract_worker(
    dataset: str,
    subset: str,
    video_path: str,
    out_dir: str,
    frames_per_video: int,
    label: str,
    method: str
) -> list:
    """
    Extract a fixed number of faces (frames) from one video.
    Returns metadata list.
    """
    metadata = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total == 0:
        return metadata

    # select random frames up to requested count
    idxs = list(range(total))
    random.shuffle(idxs)
    picked = idxs[:frames_per_video]

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    saved = 0
    for fi in picked:
        if saved >= frames_per_video:
            break
        # apply stride if desired
        if STRIDE > 1 and (fi % STRIDE) != 0:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2))
        dets = _detector(small, 1)
        if not dets:
            continue

        # pick largest face
        best = max(dets, key=lambda r: r.width()*r.height())
        x1, y1 = best.left()*2, best.top()*2
        x2, y2 = best.right()*2, best.bottom()*2

        # pad and crop
        x1, y1 = max(0, x1-PADDING), max(0, y1-PADDING)
        x2 = min(frame.shape[1], x2+PADDING)
        y2 = min(frame.shape[0], y2+PADDING)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        face = cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        # unique filename with UUID
        uid = uuid.uuid4().hex[:8]
        out_fn = f"{base}_f{fi:06d}_{saved:03d}_{uid}.jpg"
        out_fp = os.path.join(out_dir, out_fn)
        cv2.imwrite(out_fp, face)

        metadata.append({
            "dataset": dataset,
            "subset": subset,
            "filename": out_fn,
            "video": os.path.basename(video_path),
            "frame": fi,
            "label": label,
            "method": method,
            "resolution": f"{face.shape[1]}x{face.shape[0]}"
        })
        saved += 1
    cap.release()
    return metadata

# ===== VIDEO COLLECTION =====
def collect_videos(root_dir):
    vids = []
    for dp, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(('.mp4','.avi','.mov','.mkv')):
                vids.append(os.path.join(dp, fn))
    return sorted(vids)

# ===== MAIN =====
def main():
    p = argparse.ArgumentParser(
        description="Parallel face extraction with traceable CSV metadata"
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--frames_per_video", type=int, required=True,
                   help="Number of frames to extract per video")
    p.add_argument("--ffpp_c23", help="root of FF++ c23 dataset")
    p.add_argument("--dfd_real", help="root of DFDC real videos")
    p.add_argument("--dfd_fake", help="root of DFDC fake videos")
    p.add_argument("--celebdfv1_real", help="root of CelebDF-v1 real")
    p.add_argument("--celebdfv1_fake", help="root of CelebDF-v1 fake")
    p.add_argument("--celebdfv2_real", help="root of CelebDF-v2 real")
    p.add_argument("--celebdfv2_fake", help="root of CelebDF-v2 fake")
    args = p.parse_args()

    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # build source list: (dataset, subset, vids, out_dir, label, method)
    sources = []
    if args.ffpp_c23:
        base = args.ffpp_c23
        for cat, label in [("original_sequences", "real"), ("manipulated_sequences", "fake")]:
            cat_root = os.path.join(base, cat)
            if not os.path.isdir(cat_root):
                continue
            dataset, subset = "FFPP_C23", label
            for method in os.listdir(cat_root):
                vid_dir = os.path.join(cat_root, method, "c23", "videos")
                if not os.path.isdir(vid_dir):
                    continue
                vids = collect_videos(vid_dir)
                out_sub = os.path.join(args.output_dir, dataset, subset, method)
                sources.append((dataset, subset, vids, out_sub, label, method))

    mapping = {
        "DFDC_REAL": args.dfd_real,
        "DFDC_FAKE": args.dfd_fake,
        "CELEBDFV1_REAL": args.celebdfv1_real,
        "CELEBDFV1_FAKE": args.celebdfv1_fake,
        "CELEBDFV2_REAL": args.celebdfv2_real,
        "CELEBDFV2_FAKE": args.celebdfv2_fake
    }
    for name, path in mapping.items():
        if not path:
            continue
        vids = collect_videos(path)
        label = name.split('_')[-1].lower()
        method = '' if label == 'real' else name.split('_')[0]
        dataset, subset = name, label
        out_sub = os.path.join(args.output_dir, dataset, subset)
        sources.append((dataset, subset, vids, out_sub, label, method))

    # extraction loop
    all_meta = []
    for dataset, subset, vids, out_dir, label, method in sources:
        print(f"\nProcessing {dataset}/{subset}/{method or 'real'}: {len(vids)} videos")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=worker_init) as exe:
            futures = {
                exe.submit(
                    extract_worker,
                    dataset, subset, vp, os.path.join(out_dir),
                    args.frames_per_video, label, method
                ): vp for vp in vids
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset}/{subset}"):
                try:
                    all_meta.extend(fut.result())
                except Exception as e:
                    print(f"❌ {futures[fut]}: {e}")

    # write CSV metadata
    csv_path = os.path.join(args.output_dir, 'metadata.csv')
    fieldnames = [
        'dataset', 'subset', 'filename', 'video', 'frame',
        'label', 'method', 'resolution'
    ]
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_meta:
            writer.writerow(row)
    print(f"Saved {len(all_meta)} entries → {csv_path}")

if __name__ == '__main__':
    # constants
    PADDING = 64
    OUTPUT_SIZE = (256, 256)
    STRIDE = 5
    MAX_WORKERS = max(1, os.cpu_count() - 1)
    main()
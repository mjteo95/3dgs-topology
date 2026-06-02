"""
STAGE 0:        Extracts frames from input videos
STAGE 1-3:      COLMAP sparse reconstruction
STAGE 4:        Undistortion for 3DGS training
STAGE 5:        Optional binary-to-text conversion
"""
 
from __future__ import annotations
 
import logging
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Optional
 
# --- CONFIGURATION (START) ---
# FOLDER/FILE PATHS
BASE_DIR     = Path(__file__).resolve().parent
VIDEOS_DIR   = BASE_DIR.parent.parent / "data" / "videos"
OUTPUT_DIR   = BASE_DIR.parent.parent / "data" / "sparse_point_clouds"
FFMPEG       = BASE_DIR.parent / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
COLMAP       = BASE_DIR.parent / "tools" / "colmap" / "bin" / "COLMAP.exe"
VOCAB_TREE   = BASE_DIR.parent / "tools" / "colmap" / "vocab_tree" / "vocab_tree_faiss_flickr100K_words256K.bin"

# DEBUGGING / TESTING
STAGE_FRAME_EXTRACTION      = 0
STAGE_FEATURE_EXTRACTION    = 1
STAGE_FEATURE_MATCHING      = 2
STAGE_RECONSTRUCTION        = 3
STAGE_IMAGE_UNDISTORTER     = 4
STAGE_BIN_TO_TXT            = 5

# SINGLE_VIDEO: Optional[Path]    = VIDEOS_DIR / "0807.mp4"
SINGLE_VIDEO: Optional[Path]    = None
START_STAGE                     = 1

STOP_AT_FRAME_EXTRACTION        = False
STOP_AT_FEATURE_EXTRACTION      = False
STOP_AT_FEATURE_MATCHING        = False

# FFMPEG 
FRAME_FPS               = 1.5
SCENE_THRESHOLD         = 0.06
FRAME_QUALITY           = 2
 
# COLMAP / FEATURE EXTRACTION
 
# SIFT_MAX_FEATURES     = 8192   # 0470 and 0450
SIFT_MAX_FEATURES       = 4096   # 0807
SIFT_PEAK_THRESHOLD     = 0.006  # 0807
# SIFT_PEAK_THRESHOLD   = 0.003  # 0470 and 0450
SIFT_EDGE_THRESHOLD     = 10 
MIN_KEYPOINTS           = 1000   # 0807
# MIN_KEYPOINTS         = 500    # 0470 and 0450
 
# COLMAP / SEQUENTIAL FEATURE MATCHING
 
# SEQ_OVERLAP    = 20    # 0470 and 0450
SEQ_OVERLAP      = 8     # 0807
LOOP_DETECTION   = True
GUIDED_MATCHING  = True
 
# COLMAP / BUNDLE ADJUSTMENT
 
BA_REFINE_FOCAL      = 1
BA_REFINE_PRINCIPAL  = 0
BA_REFINE_EXTRA      = 1
BA_GLOBAL_ITERATIONS = 100
 
# Prints subcommands to terminal if set to True
DEBUG_COMMANDS = False
 
logging.basicConfig(
    level=logging.DEBUG if DEBUG_COMMANDS else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    filename="logfile.log"
)
log = logging.getLogger("pipeline")

# --- CONFIGURATION (END) --- 
 
def run(cmd: list, cwd: Optional[Path] = None) -> None:
    str_cmd = [str(c) for c in cmd]
    if DEBUG_COMMANDS:
        log.debug("CMD: %s", " ".join(str_cmd))
    subprocess.run(str_cmd, cwd=str(cwd) if cwd else None, check=True)
 
 
def validate_paths() -> None:
    errors = []
 
    if not FFMPEG.exists():
        errors.append(f"FFmpeg not found: {FFMPEG}")
    if not COLMAP.exists():
        errors.append(f"COLMAP not found: {COLMAP}")
    if SINGLE_VIDEO is not None and not SINGLE_VIDEO.exists():
        errors.append(f"Video not found:          {SINGLE_VIDEO}")
    elif SINGLE_VIDEO is None and not VIDEOS_DIR.exists():
        errors.append(f"Videos folder not found:  {VIDEOS_DIR}")
    if not VOCAB_TREE.exists():
        errors.append(f"Vocab tree not found: {VOCAB_TREE}")
 
    if errors:
        for e in errors:
            log.error("  %s", e)
        sys.exit(1)
 
 
def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() == ".jpg")
 
 
# --- FRAME EXTRACTION ---
 
def extract_frames(video_path: Path, frames_dir: Path) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
 
    vf = (
        f"fps={FRAME_FPS},"
        # Compares pixel blocks between consecutive frames, drops frames that have no significant block differences
        f"mpdecimate=hi=64*12:lo=64*5:frac=0.33,"
    )
 
    log.info(f"[FFMPEG] Extracting frames from: {video_path.name}")
    run([
        FFMPEG, "-y",
        "-i", video_path,
        "-vf", vf,
        "-qscale:v", FRAME_QUALITY,
        "-vsync", "vfr",
        frames_dir / "%06d.jpg",
    ])
 
    n = count_images(frames_dir)
    log.info(f"[FFMPEG] Extracted {n} frames to {frames_dir}.")
 
    if n < 200:
        log.warning(f"[FFMPEG] Only {n} frames extracted.")
    return n
 
 
# --- FEATURE EXTRACTION (COLMAP) ---
 
def feature_extraction(database: Path, images_dir: Path) -> None:
    log.info("[COLMAP] Feature extraction …")
    run([
        COLMAP, "feature_extractor",
        "--database_path",                          database,
        "--image_path",                             images_dir,
        "--ImageReader.single_camera",              "1",
        "--ImageReader.camera_model",               "SIMPLE_RADIAL",
        "--SiftExtraction.max_num_features",        SIFT_MAX_FEATURES,
        "--SiftExtraction.peak_threshold",          SIFT_PEAK_THRESHOLD,
        "--SiftExtraction.edge_threshold",          SIFT_EDGE_THRESHOLD,
        "--SiftExtraction.estimate_affine_shape",   "1",
        "--SiftExtraction.domain_size_pooling",     "1",
        "--FeatureExtraction.use_gpu",              "1",
    ])
 
 
# --- IMAGE PRUNING ---
 
def prune_images_keypoints(database: Path, frames_dir: Path) -> list[str]:
    log.info(f"[PRUNING] Removing images with <{MIN_KEYPOINTS} keypoints")
 
    con = sqlite3.connect(database)
    rows = con.execute(
        """
        SELECT i.image_id, i.name, COALESCE(k.rows, 0) AS kp_count
        FROM   images AS i
        LEFT JOIN keypoints AS k ON i.image_id = k.image_id
        WHERE  COALESCE(k.rows, 0) < ?
        ORDER BY kp_count ASC
        """,
        (MIN_KEYPOINTS,),
    ).fetchall()
    con.close()
 
    removed = []
    for _, name, kp_count in rows:
        img_path = frames_dir / name
        if img_path.exists():
            log.info(f"Removed {name} with {kp_count} keypoints.")
            img_path.unlink()
            removed.append(name)
 
    log.info(f"[PRUNING] Removed {len(removed)} image(s).")
    return removed

def prune_images_unmatched(database: Path, frames_dir: Path) -> list[str]:
    log.info("[PRUNING] Removing images with no matches.")
 
    con = sqlite3.connect(database)
 
    rows = con.execute(
        """
        SELECT image_id, name
        FROM   images
        WHERE  image_id NOT IN (
            SELECT pair_id / 2147483647 FROM two_view_geometries WHERE rows > 0
            UNION
            SELECT pair_id % 2147483647 FROM two_view_geometries WHERE rows > 0
        )
        ORDER BY name ASC
        """
    ).fetchall() # pair_id = id1 * 2147483647 (max signed 32-bit integer) + id2
 
    removed = []
    for image_id, name in rows:
        # Remove from database tables
        con.execute("DELETE FROM keypoints   WHERE image_id = ?", (image_id,))
        con.execute("DELETE FROM descriptors WHERE image_id = ?", (image_id,))
        con.execute("DELETE FROM images      WHERE image_id = ?", (image_id,))
 
        # Remove from disk
        img_path = frames_dir / name
        if img_path.exists():
            img_path.unlink()
        removed.append(name)
 
    con.commit()
    con.close()
 
    log.info(f"[PRUNING] Removed {len(removed)} unmatched image(s).")
    return removed
 
 
def reset_database(database: Path) -> None:
    if database.exists():
        database.unlink()
        log.info("[DATABASE] Reset database.")
 
 
# --- SEQUENTIAL FEATURE MATCHING ---
 
def sequential_matching(database: Path) -> None:
    log.info("[COLMAP] Starting sequential matching.")
 
    cmd = [
        COLMAP, "sequential_matcher",
        "--database_path", database,
        "--SequentialMatching.overlap", SEQ_OVERLAP,
        "--SequentialMatching.loop_detection", "1",
        "--FeatureMatching.guided_matching", "1" if GUIDED_MATCHING else "0",
        "--SequentialMatching.loop_detection_num_images", "50",
        "--SequentialMatching.vocab_tree_path", VOCAB_TREE,
    ]
 
    run(cmd)
 
 
# --- RECONSTRUCTION ---
 
def mapper(database: Path, images_dir: Path, sparse_dir: Path) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    log.info("[COLMAP] Incremental mapper …")
 
    run([
        COLMAP, "mapper",
        "--database_path",                       database,
        "--image_path",                          images_dir,
        "--output_path",                         sparse_dir,
        "--Mapper.ba_refine_focal_length",       BA_REFINE_FOCAL,
        "--Mapper.ba_refine_principal_point",    BA_REFINE_PRINCIPAL,
        "--Mapper.ba_refine_extra_params",       BA_REFINE_EXTRA,
        "--Mapper.ba_global_max_num_iterations", BA_GLOBAL_ITERATIONS,
        "--Mapper.ba_local_max_num_iterations",  "25",
        "--Mapper.min_num_matches",              "15",
        "--Mapper.init_min_num_inliers",         "50",
        "--Mapper.multiple_models",              "0", # For single unified model to better represent room
    ])

# --- EXPORT ---
def export_sparse(sparse_dir: Path, export_dir: Path, video_name: str) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
 
    expected_files = {"cameras.bin", "images.bin", "points3D.bin"}
    if not expected_files.issubset({f.name for f in sparse_dir.iterdir()}):
        log.warning(f"[EXPORT] Model files not found in {sparse_dir}.")
        return
 
    run([
        COLMAP, "model_converter",
        "--input_path",  sparse_dir,
        "--output_path", sparse_dir,
        "--output_type", "TXT",
    ])
    log.info(f"[EXPORT] TXT version written to: {sparse_dir}.")

    ply_path = export_dir / f"{video_name}_sparse.ply"
    run([
        COLMAP, "model_converter",
        "--input_path",  sparse_dir,
        "--output_path", ply_path,
        "--output_type", "PLY",
    ])
    log.info(f"[EXPORT] PLY written to: {ply_path}.")

# --- IMAGE UNDISTORTION (LICHTFELD STUDIO PREP) ---
 
def image_undistorter(sparse_dir: Path, frames_dir: Path, undistorted_dir: Path) -> None:
    shutil.rmtree(undistorted_dir, ignore_errors=True)
    undistorted_dir.mkdir(parents=True, exist_ok=True)
 
    log.info("[COLMAP] Running image undistorter ...")
    run([
        COLMAP, "image_undistorter",
        "--image_path",   frames_dir,
        "--input_path",   sparse_dir / "0",
        "--output_path",  undistorted_dir,
        "--output_type",  "COLMAP",
    ])
    log.info("[COLMAP] Image undistortion complete.")
 
 
def restructure_for_lichtfeld(undistorted_dir: Path, lichtfeld_dir: Path) -> None:
    """
    Restructures COLMAP's undistorter output to match Lichtfeld Studio's
    required input layout:
 
        lichtfeld/
        ├── images/ 
        └── sparse/
    """
    shutil.rmtree(lichtfeld_dir, ignore_errors=True)
    lichtfeld_dir.mkdir(parents=True, exist_ok=True)
 
    src_images = undistorted_dir / "images"
    dst_images = lichtfeld_dir / "images"
    log.info("[RESTRUCTURE] Copying undistorted images ...")
    shutil.copytree(src_images, dst_images)
 
    src_sparse = undistorted_dir / "sparse"
    dst_sparse_0 = lichtfeld_dir / "sparse" / "0"
    dst_sparse_0.mkdir(parents=True, exist_ok=True)
 
    model_files = {"cameras.bin", "images.bin", "points3D.bin"}
    found = {f.name for f in src_sparse.iterdir() if f.is_file()}
    missing = model_files - found
    if missing:
        log.warning(f"[RESTRUCTURE] Undistorted model is missing files: {missing}.")
        return
 
    for filename in model_files:
        shutil.copy2(src_sparse / filename, dst_sparse_0 / filename)
 
    log.info(f"[RESTRUCTURE] Folder structured for Lichtfeld Studio at: {lichtfeld_dir}.")
 
# --- PER-VIDEO FUNCTION ---
def process_video(video_path: Path) -> None:
    name = video_path.stem
    log.info("=" * 70)
    log.info("Processing: %s", video_path.name)
    log.info("=" * 70)

    stage = START_STAGE
 
    # Workspace layout
    work        = OUTPUT_DIR / name
    frames      = work / "frames"
    colmap_w    = work / "colmap"
    database    = colmap_w / "database.db"
    sparse      = colmap_w / "sparse"
    export      = work / "export"
    undistorted = work / "undistorted"
    lichtfeld   = work / "lichtfeld"
 
    colmap_w.mkdir(parents=True, exist_ok=True)
 
    # --- FRAME EXTRACTION ---
    if stage > STAGE_FRAME_EXTRACTION and count_images(frames) > 0:
        log.info(f"[FFMPEG] Existing frames found, skipping extraction.")
    else:
        shutil.rmtree(frames, ignore_errors=True)
        n_frames = extract_frames(video_path, frames)
        if n_frames < 200:
            log.error(f"[FFMPEG] Too few frames extracted ({n_frames}) for {name}. Consider changing parameters.")
            return
    if STOP_AT_FRAME_EXTRACTION:
        return
 
    # --- FEATURE EXTRACTION (1) ---
    if stage <= STAGE_FEATURE_EXTRACTION:
        reset_database(database)
        feature_extraction(database, frames)
 
    # --- IMAGE PRUNING ---
        removed = prune_images_keypoints(database, frames)
    
    # --- FEATURE EXTRACTION (2) ---
        if removed:
            reset_database(database)
            feature_extraction(database, frames)
            removed_2 = prune_images_keypoints(database, frames)

            # FEATURE EXTRACTION (3) - If necessary
            if removed_2:
                reset_database(database)
                feature_extraction(database, frames)
        if STOP_AT_FEATURE_EXTRACTION:
            return

    # --- SEQUENTIAL MATCHING ---
    if stage <= STAGE_FEATURE_MATCHING:
        sequential_matching(database)
        if STOP_AT_FEATURE_MATCHING:
            return
        prune_images_unmatched(database, frames)
    
    # --- RECONSTRUCTION ---
    if stage <= STAGE_RECONSTRUCTION:
        shutil.rmtree(sparse, ignore_errors=True)
        mapper(database, frames, sparse)
 
    # --- EXPORT ---
    export_sparse(sparse, export, name)

    # --- IMAGE UNDISTORTION ---
    if stage <= STAGE_IMAGE_UNDISTORTER:
        image_undistorter(sparse, frames, undistorted)
 
    # --- RESTRUCTURE FOR LICHTFELD STUDIO ---
        restructure_for_lichtfeld(undistorted, lichtfeld)
    
    # --- CONVERT BIN TO TXT FILES ---
    if stage <= STAGE_BIN_TO_TXT:
        export_sparse(undistorted / "sparse", undistorted / "sparse", name)
 
    log.info(f"[END] Video {name} processed to {work}.")
 
 
# --- MAIN ---
 
def main() -> None:
    validate_paths()
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if SINGLE_VIDEO is not None:
        videos = [SINGLE_VIDEO]
    else: 
        videos = sorted(p for p in VIDEOS_DIR.iterdir())
    if not videos:    
        log.error(f"No videos found in {VIDEOS_DIR}.")
        sys.exit(1)
 
    log.info(f"{len(videos)} video(s) to process.")
 
    failed = []
    for video in videos:
        try:
            process_video(video)
        except subprocess.CalledProcessError as e:
            log.error(f"[ERROR] {video.name} → subprocess exited with code {e.returncode}.")
            failed.append(video.name)
        except Exception as e:
            log.exception(f"[ERROR] {video.name}: {e}.")
            failed.append(video.name)
    
    num_failed = len(failed)
    num_passed = len(videos) - num_failed
    log.info("=" * 70)
    log.info(f"Finished: {num_passed} succeeded, {num_failed} failed.")
    if failed:
        log.warning("Failed videos: %s", ", ".join(failed))
 
if __name__ == "__main__":
    main()
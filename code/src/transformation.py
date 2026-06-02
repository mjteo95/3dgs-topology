from __future__ import annotations
import json
import struct
import numpy as np
import psycopg2
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import open3d as o3d

# --- CONFIGURATION ---
BASE_DIR    = Path(__file__).resolve().parent
OUTPUT_DIR  = BASE_DIR.parent.parent / "data"
DATA_JSON   = OUTPUT_DIR / "floor_plan_data.json"

SYNTHETIC_IFC_PLY = OUTPUT_DIR / "0470_synthetic_pc.ply" # For visualisation only
DXF_UNIT_TO_M = 0.001

# 3D scatter plot of reference versus target. Per cell.
ENABLE_ALIGNMENT_PLOT = False
# Save the transformed sparse cloud as a PLY alongside the points3d file.
# NOTE: For visualisation and outputting of transformed sparse reconstructions, 
# You must provide and replace the relative file paths in the input JSON with valid points3d.bin/txt paths.
SAVE_ALIGNED_PLY = False
# For alignment visualisation.
# Requires SAVE_ALIGNED_PLY = True and a valid SYNTHETIC_IFC_PLY path (or any reference point cloud).
ENABLE_OPEN3D_VIS = False

# Per-cell correspondence table and RMS report, saved to OUTPUT_DIR/transformation_report.txt.
WRITE_REPORT = True

# --- DATABASE CONNECTION ---

DB_PARAMS = {
    "host":   "localhost",
    "port":   5432,
    "dbname": "topology",
    "user":   "postgres",
    # Password omitted — uses ~/.pgpass
}

# --- LOAD JSON ---

def has_null(value) -> bool:
    """ Checks for null values or empty strings for the point coordinates.
        Used to flag and remove null corresponding point pairs for the Umeyama transformation calculation.
    """
    if value is None:           return True
    if isinstance(value, str):  return value.strip() == ""
    if isinstance(value, list): return any(has_null(v) for v in value)
    return False


def load_floor_plan_data(json_path: Path) -> dict[str, dict]:
    """
    Reads floor_plan_data.json, and returns a nested dictionary of all Cells.
    Each Cell has a {cell_id : cell_dict}, which contains:

        points3d_path : Path          full path to the cell's points3d file
        dwg           : ndarray (N,3) DWG feature points
        colmap        : ndarray (N,3) COLMAP XYZ
        n_skipped     : int           number of null point-pairs
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {json_path}\n"
            "Fill in floor_plan_data.json before running this script."
        )
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse {json_path.name}: {exc}") from exc

    cells: dict[str, dict] = {}
    warns: list[str] = []

    for cell_id, entry in data.get("umeyama", {}).items():
        if not isinstance(entry, dict):
            continue

        p3d_rel = entry.get("points3d_file")
        # if not p3d_rel:
        #     warns.append(f"  {cell_id}: 'points3d_file' missing — skipping")
        #     continue
        p3d_path = BASE_DIR / p3d_rel
        # if not p3d_path.exists():
        #     alt = p3d_path.with_suffix(".bin")
        #     if alt.exists():
        #         p3d_path = alt
        #     else:
        #         warns.append(f"  {cell_id}: points3d_file not found: {p3d_path} — skipping")
        #         continue

        dwg_raw    = entry.get("dwg_vertices",  [])
        colmap_raw = entry.get("sparse_points", [])

        if not dwg_raw or not colmap_raw:
            warns.append(f"  {cell_id}: dwg_vertices or sparse_points missing — skipping")
            continue
        if len(dwg_raw) != len(colmap_raw):
            warns.append(
                f"  {cell_id}: dwg_vertices ({len(dwg_raw)}) and "
                f"sparse_points ({len(colmap_raw)}) lengths differ — skipping"
            )
            continue

        valid_dwg:    list = []
        valid_colmap: list = []
        n_skipped = 0

        for dwg_pt, colmap_pt in zip(dwg_raw, colmap_raw):
            if colmap_pt is None or has_null(colmap_pt):
                n_skipped += 1
                continue
            if has_null(dwg_pt):
                warns.append(
                    f"  {cell_id}: null dwg_vertex paired with a valid "
                    f"sparse_point — pair skipped (fix dwg_vertices)"
                )
                n_skipped += 1
                continue
            valid_dwg.append(dwg_pt)
            valid_colmap.append(colmap_pt)

        if len(valid_dwg) < 3:
            warns.append(
                f"  {cell_id}: only {len(valid_dwg)} valid pair(s) after "
                f"filtering ({n_skipped} null) — need at least 3, skipping"
            )
            continue

        try:
            
            # Convert from mm to m, then pad to (N, 3) with Z=0
            dwg_arr = np.array(valid_dwg, dtype=float) * DXF_UNIT_TO_M
            if dwg_arr.ndim == 2 and dwg_arr.shape[1] == 2:
                dwg_arr = np.column_stack([dwg_arr, np.zeros(len(dwg_arr))])
            cells[cell_id] = {
                "points3d_path": p3d_path,
                "dwg":           dwg_arr,
                "colmap":        np.array(valid_colmap, dtype=float),
                "n_skipped":     n_skipped,
            }
        except ValueError as exc:
            warns.append(f"  {cell_id}: bad point format — {exc}")

    if warns:
        print("Warnings / skipped cells:")
        print("\n".join(warns))
    if not cells:
        raise ValueError(
            f"No usable umeyama cells in {json_path.name}. "
            "Each cell needs a valid points3d_file and at least 3 non-null sparse_points."
        )
    print(f"Loaded {len(cells)} umeyama cell(s): {list(cells)}\n")
    return cells

# --- LOAD POINTS3D ---
# This is only for visualisation
def read_points3d_txt(path: Path) -> np.ndarray:
    raw = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.strip().split()
            if len(p) >= 4:
                raw.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(raw, dtype=float)


def read_points3d_bin(path: Path) -> np.ndarray:
    raw = []
    with open(path, "rb") as fh:
        num_points = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(num_points):
            fh.read(8)                            # point3D_id  (uint64)
            x, y, z = struct.unpack("<ddd", fh.read(24))
            fh.read(3)                            # rgb         (3 x uint8)
            fh.read(8)                            # error       (float64)
            track_len = struct.unpack("<I", fh.read(4))[0]
            fh.read(track_len * 8)                # track       (track_len x 2 x uint32)
            raw.append([x, y, z])
    return np.array(raw, dtype=float)


def load_points3d(path: Path) -> np.ndarray:
    """Load a COLMAP points3D file (.txt or .bin) and return an (N, 3) XYZ array."""
    if path.suffix == ".bin":
        return read_points3d_bin(path)
    return read_points3d_txt(path)

# --- TRANSFORMATION ---

def floor_normal_from_correspondences(colmap_pts_permuted: np.ndarray) -> np.ndarray:
    """
    Estimate the floor plane normal from the COLMAP correspondence points.
    This assumes the selected points are coplanar, and the plane is equivalent to the DWG floor plane.
    """
    pts = colmap_pts_permuted
    _, _, Vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    normal = Vt[-1]  # smallest singular value = plane normal
    if normal[2] < 0:
        normal = -normal
    return normal / np.linalg.norm(normal)


def align_to_z(normal: np.ndarray) -> np.ndarray:
    """Aligns floor plane normal to Z-axis [0,0,1] using the Rodrigues rotation formula"""
    if normal[2] < 0:
        normal = -normal
    v = np.cross(normal, np.array([0., 0., 1.])) # rotation axis
    s = np.linalg.norm(v)
    c = float(np.dot(normal, np.array([0., 0., 1.])))
    if s < 1e-10:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / s**2


def umeyama_2d(P2d: np.ndarray, Q2d: np.ndarray):
    """
    2-D similarity transform (scale + rotation, no reflection).
    Tries all four axis-sign combinations and returns the one with the lowest RMS.
    Returns (scale, R2d_eff, t2d_eff, P_aligned, rms).
    """
    n    = len(P2d)
    mu_q = Q2d.mean(0)
    Qc   = Q2d - mu_q
    best = None
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]: # axis permutation
        Pf   = P2d * np.array([float(sx), float(sy)])
        mu_p = Pf.mean(0)
        varP = np.var(Pf, axis=0).sum()
        C    = (Qc.T @ (Pf - mu_p)) / n # cross-covariance matrix of P and Q
        U, S, Vt = np.linalg.svd(C) 
        if (np.linalg.det(U) * np.linalg.det(Vt)) < 0:
            continue
        R2d   = U @ Vt
        scale = float((S[0] + S[1]) / varP)
        t2d   = mu_q - scale * R2d @ mu_p
        Pt    = scale * (R2d @ Pf.T).T + t2d
        rms   = float(np.sqrt(np.mean(np.sum((Pt - Q2d) ** 2, axis=1))))
        if best is None or rms < best[-1]:
            R2d_eff = R2d @ np.diag([float(sx), float(sy)])
            t2d_eff = mu_q - scale * R2d_eff @ P2d.mean(0)
            best = (scale, R2d_eff, t2d_eff, Pt, rms)
    if best is None:
        raise RuntimeError("No proper-rotation 2-D solution found.")
    return best

# --- PERMUTATION FUNCTIONS ---

# All 6 orderings of the three COLMAP axes tried during the permutation search.
ALL_PERMS = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]

def run_with_perm(
    perm:       tuple[int, int, int],
    colmap_pts: np.ndarray,
    dwg_pts:    np.ndarray,
) -> dict | None:
    
    P_perm = colmap_pts[:, perm]

    floor_normal = floor_normal_from_correspondences(P_perm)
    R_level      = align_to_z(floor_normal)
    P_lev        = (R_level @ P_perm.T).T
    z_floor      = P_lev[:, 2].mean()
    tilt         = float(np.degrees(np.arccos(np.clip(floor_normal[2], 0, 1))))

    try:
        scale, R2d_eff, t2d_eff, _, rms_xy = umeyama_2d(P_lev[:, :2], dwg_pts[:, :2])
    except RuntimeError:
        return None

    r = R2d_eff.flatten()
    T2d = np.array([
        [scale*r[0], scale*r[1], 0.,     t2d_eff[0]     ],
        [scale*r[2], scale*r[3], 0.,     t2d_eff[1]     ],
        [0.,         0.,         -scale, scale * z_floor ],
        [0.,         0.,         0.,     1.              ],
    ])
    T_level         = np.eye(4)
    T_level[:3, :3] = R_level
    T = T2d @ T_level

    P_h     = np.column_stack([P_perm, np.ones(len(P_perm))])
    P_t     = (T @ P_h.T).T[:, :3]
    errs_xy = np.linalg.norm(P_t[:, :2] - dwg_pts[:, :2], axis=1)
    errs_z  = np.abs(P_t[:, 2]          - dwg_pts[:, 2])

    return {
        "T":            T,
        "rms_xy":       float(rms_xy),
        "rms_z":        float(errs_z.mean()),
        "rms_3d":       float(np.sqrt(np.mean(np.sum((P_t - dwg_pts) ** 2, axis=1)))),
        "errs_xy":      errs_xy,
        "errs_z":       errs_z,
        "scale":        float(scale),
        "floor_normal": floor_normal,
        "tilt":         tilt,
        "z_floor":      float(z_floor),
        "P_t":          P_t,
        "perm":         perm,
    }

# --- PER-CELL ---

def run_cell_pipeline(cell_id: str, cell: dict) -> tuple[np.ndarray, dict]:
    dwg_pts    = cell["dwg"]            # (N, 3) in metres
    colmap_pts = cell["colmap"]         # (N, 3) raw COLMAP XYZ
    p3d_path   = cell["points3d_path"]

    print(f"  Searching {len(ALL_PERMS)} axis permutations ...")
    search: dict[tuple, dict] = {}
    for perm in ALL_PERMS:
        r = run_with_perm(perm, colmap_pts, dwg_pts)
        if r is not None:
            search[perm] = r

    if not search:
        raise RuntimeError(
            "All 6 axis permutations failed. "
            "Check the points3d file and correspondence data."
        )

    print(f"  {'Perm':>10}  {'XY-RMS m':>10}  {'Z-RMS m':>8}  {'Tilt':>6}")
    for perm, r in sorted(search.items(), key=lambda kv: kv[1]["rms_xy"]):
        print(f"  {str(perm):>10}  {r['rms_xy']:>10.4f}  {r['rms_z']:>8.4f}  "
                f"{r['tilt']:>5.1f} deg")

    best_perm = min(search, key=lambda p: search[p]["rms_xy"])
    print(f"  Best perm: {best_perm}  XY-RMS {search[best_perm]['rms_xy']:.4f} m")
    result = search[best_perm]

    T = result["T"]
    print(f"  Floor normal {result['floor_normal'].round(4)}  tilt {result['tilt']:.2f} deg")
    print(f"  Scale {result['scale']:.6f}  XY-RMS {result['rms_xy']:.4f} m "
          f"({result['rms_xy']*100:.1f} cm)")
    print(f"  RMS  3D={result['rms_3d']:.4f} m   "
          f"XY={result['rms_xy']:.4f} m   Z={result['rms_z']:.4f} m")

    corr = {
        "dwg_pts":   dwg_pts,
        "P_t":       result["P_t"],
        "errs_xy":   result["errs_xy"],
        "errs_z":    result["errs_z"],
        "rms_xy":    result["rms_xy"],
        "rms_z":     result["rms_z"],
        "rms_3d":    result["rms_3d"],
        "scale":     result["scale"],
        "tilt":      result["tilt"],
        "n_valid":   len(dwg_pts),
        "n_skipped": cell["n_skipped"],
        "T":         T,
        "p3d_path":  p3d_path,
        "perm":      best_perm,
    }
    return T, corr

# --- VISUALISATION ---

def show_alignment_plot(cell_id: str, corr: dict) -> None:
    dwg = corr["dwg_pts"]
    P_t = corr["P_t"]
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(dwg[:, 0], dwg[:, 1], dwg[:, 2],
               c="green", s=100, label="DWG reference", marker="^")
    for i in range(len(dwg)):
        ax.text(dwg[i, 0], dwg[i, 1], dwg[i, 2],
                f"A{i}", fontsize=10, color="darkgreen")
    ax.scatter(P_t[:, 0], P_t[:, 1], P_t[:, 2],
               c="red", s=100, label="COLMAP (transformed)", marker="o")
    for i in range(len(P_t)):
        ax.text(P_t[i, 0], P_t[i, 1], P_t[i, 2],
                f"{i}", fontsize=10, color="darkred")
    ax.legend()
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    plt.title(f"Cell {cell_id}  XY-RMS={corr['rms_xy']*100:.1f} cm")
    plt.show()


def save_aligned_ply(cell_id: str, corr: dict) -> Path | None:
    T        = corr["T"]
    p3d_path = corr["p3d_path"]
    pts_all  = load_points3d(p3d_path)
    pts_perm = pts_all[:, corr["perm"]]
    h_pts    = np.column_stack([pts_perm, np.ones(len(pts_perm))])
    pts_out  = (T @ h_pts.T).T[:, :3]

    ply_path = p3d_path.parent / f"{cell_id}_aligned.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_out)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"  Saved {len(pts_out)} points to {ply_path}")
    return ply_path


def show_open3d_vis(cell_id: str, ply_path: Path) -> None:
    if not SYNTHETIC_IFC_PLY.exists():
        print(f"  SYNTHETIC_IFC_PLY not found ({SYNTHETIC_IFC_PLY}) — skipping")
        return
    source     = o3d.io.read_point_cloud(str(ply_path))
    target_vis = o3d.io.read_point_cloud(str(SYNTHETIC_IFC_PLY))
    source.paint_uniform_color([1, 0, 0])
    target_vis.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries(
        [source, target_vis],
        window_name=f"Cell {cell_id} — COLMAP (red) vs IFC synthetic (green)",
        width=1280, height=720,
    )


def append_report(report_path: Path, cell_id: str, corr: dict) -> None:
    """Append a correspondence table and RMS summary for one cell to the report file."""
    dwg = corr["dwg_pts"]
    P_t = corr["P_t"]
    T   = corr["T"]
    lines = [
        f"\n{'='*70}",
        f"Cell {cell_id}   valid={corr['n_valid']}  skipped={corr['n_skipped']}",
        f"{'='*70}",
        f"Scale:   {corr['scale']:.6f}",
        f"Tilt:    {corr['tilt']:.2f} deg",
        f"RMS  XY: {corr['rms_xy']:.4f} m  ({corr['rms_xy']*100:.1f} cm)",
        f"RMS  Z:  {corr['rms_z']:.4f} m",
        f"RMS  3D: {corr['rms_3d']:.4f} m",
        f"Perm:    {corr['perm']}",
        "",
        f"{'Idx':>4}  {'DWG ref (m)':^36}  {'Aligned (m)':^36}  {'|XY| m':>7}  {'|Z| m':>7}",
        "-" * 96,
    ]
    for i in range(len(dwg)):
        ref_s = f"({dwg[i,0]:.3f}, {dwg[i,1]:.3f}, {dwg[i,2]:.3f})"
        trn_s = f"({P_t[i,0]:.3f}, {P_t[i,1]:.3f}, {P_t[i,2]:.3f})"
        lines.append(
            f"{i:>4}  {ref_s:^36}  {trn_s:^36}"
            f"  {corr['errs_xy'][i]:>7.4f}  {corr['errs_z'][i]:>7.4f}"
        )
    lines += ["", "4x4 Transform:"]
    for row in T:
        lines.append("  " + "  ".join(f"{v:>12.6f}" for v in row))

    with open(report_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

# ─── DATABASE UPDATE ─────────────────────────────────────────────────────────

def update_db_transforms(transforms: dict[str, dict]) -> None:
    print("\nConnecting to PostgreSQL ...")
    try:
        conn = psycopg2.connect(**DB_PARAMS)
    except psycopg2.OperationalError as exc:
        print(f"  DB connection failed: {exc}\n  Transforms NOT written to database.")
        return

    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            for cell_id, entry in transforms.items():
                T    = entry["T"]
                perm = entry["perm"]
                record = json.dumps({"matrix": T.tolist(), "perm": list(perm)})
                cur.execute(
                    "UPDATE room SET transform = %s::jsonb WHERE id = %s",
                    (record, cell_id),
                )
                if cur.rowcount == 0:
                    print(f"  WARNING: room '{cell_id}' not found in DB — skipped")
                else:
                    print(f"  room '{cell_id}': transform updated  perm={list(perm)}")
        conn.commit()
        print("Database update committed.")
    except Exception as exc:
        conn.rollback()
        print(f"DB error — rolled back: {exc}")
        raise
    finally:
        conn.close()

# --- MAIN ---

print("=== LOADING FLOOR PLAN DATA ===")
umeyama_cells = load_floor_plan_data(DATA_JSON)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
report_path = OUTPUT_DIR / "transformation_report.txt"
if WRITE_REPORT:
    report_path.write_text(
        f"Transformation Report\nData: {DATA_JSON}\n",
        encoding="utf-8",
    )

np.set_printoptions(precision=6, suppress=True)
transforms: dict[str, dict] = {}  # {cell_id: {"T": ndarray, "perm": tuple}}

for cell_id, cell in umeyama_cells.items():
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"CELL {cell_id}  {len(cell['dwg'])} valid pairs, "
          f"{cell['n_skipped']} null-skipped")
    print(sep)

    T, corr = run_cell_pipeline(cell_id, cell)
    transforms[cell_id] = {"T": T, "perm": corr["perm"]}

    print(f"\n4x4 Transform (cell {cell_id})  perm={corr['perm']}:")
    print(T)

    if WRITE_REPORT:
        append_report(report_path, cell_id, corr)
        print(f"  Report updated: {report_path}")

    if ENABLE_ALIGNMENT_PLOT:
        show_alignment_plot(cell_id, corr)

    ply_path = None
    if SAVE_ALIGNED_PLY:
        ply_path = save_aligned_ply(cell_id, corr)

    if ENABLE_OPEN3D_VIS and ply_path is not None:
        show_open3d_vis(cell_id, ply_path)

print(f"\n{'='*60}")
print(f"Transforms computed for: {list(transforms)}")

update_db_transforms(transforms)
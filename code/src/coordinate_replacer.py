from __future__ import annotations
import json
import sys
from lxml import etree
from pathlib import Path
import numpy as np

# --- CONFIG ---

DIRECTORY  = Path(__file__).resolve().parent.parent.parent
INPUT_GML  = DIRECTORY /  "data" / "manual-bk.gml"
OUTPUT_GML = DIRECTORY / "data" / "aligned-bk.gml"
DATA_JSON  = DIRECTORY / "data" / "floor_plan_data.json"

FLOOR_Z           = 0.0
DXF_UNIT_TO_M     = 0.001   # DXF millimetres → metres; applied once in the loader

# --- NAMESPACES ---

NS = {
    "gml":   "http://www.opengis.net/gml/3.2",
    "core":  "http://www.opengis.net/indoorgml/1.0/core",
    "navi":  "http://www.opengis.net/indoorgml/1.0/navigation",
    "xlink": "http://www.w3.org/1999/xlink",
    "xsi":   "http://www.w3.org/2001/XMLSchema-instance",
}
NAVI = NS["navi"]
CORE = NS["core"]
GML  = NS["gml"]

CONNECTION_TAG = f"{{{NAVI}}}ConnectionSpace"
CELL_SPACE_TAGS = [
    CONNECTION_TAG,
    f"{{{NAVI}}}GeneralSpace",
    f"{{{NAVI}}}TransitionSpace",
    f"{{{CORE}}}CellSpace",
]
BOUNDARY_TAGS = [
    f"{{{NAVI}}}ConnectionBoundary",
    f"{{{CORE}}}CellSpaceBoundary",
]

# --- PARSE JSON ---

def has_null(value) -> bool:
    if value is None:           return True
    if isinstance(value, str):  return value.strip() == ""
    if isinstance(value, list): return any(has_null(v) for v in value)
    return False


def load_dwg_vertices(json_path: Path) -> dict[str, list[tuple[float, float]]]:
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

    result:   dict[str, list[tuple[float, float]]] = {}
    warnings: list[str] = []

    for cell_id, cell_data in data.get("cells", {}).items():
        if not isinstance(cell_data, dict):
            continue
        verts = cell_data.get("room_vertices", [])
        if not verts:
            warnings.append(f"  {cell_id}: room_vertices empty — skipping")
            continue
        bad = [i for i, v in enumerate(verts) if has_null(v)]
        if bad:
            warnings.append(
                f"  {cell_id}: {len(bad)} null vertex/vertices "
                f"(indices {bad}) — skipping"
            )
            continue
        try:
            result[cell_id] = [
                (float(v[0]) * DXF_UNIT_TO_M, float(v[1]) * DXF_UNIT_TO_M)
                for v in verts
            ]
        except (TypeError, ValueError, IndexError) as exc:
            warnings.append(f"  {cell_id}: bad vertex format — {exc}")

    if warnings:
        print("Cells skipped (incomplete room_vertices):")
        print("\n".join(warnings))
    if not result:
        raise ValueError(
            f"No usable cell data in {json_path.name}. "
        )
    print(f"Loaded {len(result)} cell(s): {list(result)}")
    return result


# --- GEOMETRY ---

def find_all_cells(root):
    cells = []
    for tag in CELL_SPACE_TAGS:
        cells.extend(root.findall(f".//{tag}"))
    return cells

def find_all_boundaries(root):
    boundaries = []
    for tag in BOUNDARY_TAGS:
        boundaries.extend(root.findall(f".//{tag}"))
    return boundaries

def get_cell_type(element) -> str:
    tag = element.tag
    return tag.split("}")[1] if "}" in tag else tag

def parse_pos(text: str) -> tuple[float, float, float]:
    parts = list(map(float, text.strip().split()))
    return (parts[0], parts[1], parts[2] if len(parts) > 2 else 0.0)

def fmt_pos(x, y, z) -> str:
    return f"{x} {y} {z}"

def ring_to_pts(ring_el) -> list[tuple[float, float, float]]:
    return [parse_pos(p.text)
            for p in ring_el.findall("gml:pos", NS)
            if p.text and p.text.strip()]

def ring_xy(ring_el) -> list[tuple[float, float]]:
    return [(x, y) for x, y, z in ring_to_pts(ring_el)]

def is_floor_ring(ring_el, tol: float = 0.1) -> bool:
    return all(abs(parse_pos(p.text)[2] - FLOOR_Z) < tol
               for p in ring_el.findall("gml:pos", NS)
               if p.text and p.text.strip())

def centroid(verts: list[tuple[float, float]]) -> tuple[float, float]:
    xs, ys = zip(*verts)
    return sum(xs) / len(xs), sum(ys) / len(ys)

def nearest_key(lookup, x, y, tol=0.05):
    best_d, best_k = tol * tol, None
    for kx, ky in lookup:
        d = (kx - x) ** 2 + (ky - y) ** 2
        if d < best_d:
            best_d, best_k = d, (kx, ky)
    return best_k

# --- GLOBAL LOOKUP ---

def build_xy_lookup(
    cid: str,
    gml_verts: list[tuple[float, float]],
    dwg_verts: list[tuple[float, float]],
) -> dict[tuple[float, float], tuple[float, float]]:
    """
    Input vertices of both lists could have different ordering.
    Sequentially rotates the order of input vertices (and flip). 
    The order with the lowest alignment error is used to correspond point-pairs, which are fed into the global lookup.
    """
    n = len(gml_verts)
    assert len(dwg_verts) == n, (
        f"[{cid}] Vertex count mismatch: GML has {n}, "
        f"JSON room_vertices has {len(dwg_verts)}"
    )

    gml = np.array(gml_verts, float)
    dwg = np.array(dwg_verts, float)

    gml_c = gml - gml.mean(0)
    dwg_c = dwg - dwg.mean(0)
    gs = gml_c / (np.sqrt((gml_c ** 2).sum(1).mean()) or 1)
    ds = dwg_c / (np.sqrt((dwg_c ** 2).sum(1).mean()) or 1)

    best_err, best_off, best_flip = float("inf"), 0, False
    for flip in (False, True):
        arr = ds if not flip else ds[::-1]
        for off in range(n):
            err = float(np.sum((gs - np.roll(arr, off, 0)) ** 2))
            if err < best_err:
                best_err, best_off, best_flip = err, off, flip

    ordered = dwg[::-1] if best_flip else dwg.copy()
    ordered = np.roll(ordered, best_off, 0)
    return {(gx, gy): (float(dx), float(dy))
            for (gx, gy), (dx, dy) in zip(gml_verts, ordered)}


def build_global_lookup(
    root,
    dwg_room_vertices: dict[str, list[tuple[float, float]]],
) -> dict[tuple[float, float], tuple[float, float]]:
    """
    Door Cells are first aligned and points mapped.
    Then, the common points between Doors and Rooms/Corridor can be skipped.
    Remaining points of Room/Corridor Cells are mapped against room_vertices in the JSON.
    """
    lookup: dict[tuple[float, float], tuple[float, float]] = {}

    def get_floor_ring(cs):
        for ring in cs.findall(".//gml:LinearRing", NS):
            if is_floor_ring(ring):
                v = ring_xy(ring)
                if v and v[0] == v[-1]:
                    v = v[:-1]
                return v
        return []

    print("Phase 1 — mapping door cells (ConnectionSpace) …")
    for cs in find_all_cells(root):
        if cs.tag != CONNECTION_TAG:
            continue
        cid = cs.get(f"{{{GML}}}id")
        if cid not in dwg_room_vertices:
            print(f"  [{cid}] ConnectionSpace: not in JSON — skipping.")
            continue
        gml_v = get_floor_ring(cs)
        if not gml_v:
            continue
        try:
            lk = build_xy_lookup(cid, gml_v, dwg_room_vertices[cid])
            lookup.update(lk)
            print(f"  [{cid}] ConnectionSpace: mapped {len(lk)} vertices")
        except AssertionError as exc:
            print(f"  ERROR: {exc}")


    print("Phase 2 — mapping room / corridor cells …")
    for cs in find_all_cells(root):
        if cs.tag == CONNECTION_TAG:
            continue
        cid       = cs.get(f"{{{GML}}}id")
        cell_type = get_cell_type(cs)
        if cid not in dwg_room_vertices:
            print(f"  [{cid}] {cell_type}: not in JSON — skipping.")
            continue
        gml_v = get_floor_ring(cs)
        if not gml_v:
            continue

        unmatched: list[tuple[float, float]] = []
        for x, y in gml_v:
            if nearest_key(lookup, x, y, tol=0.05) is not None:
                continue                                  # shared vertex already mapped
            unmatched.append((x, y))

        expected = len(dwg_room_vertices[cid])
        if len(unmatched) != expected:
            print(
                f"  [{cid}] {cell_type}: {len(unmatched)} unmatched GML vertices "
                f"but room_vertices in JSON has {expected}. "
                f"Update room_vertices count to {len(unmatched)} to match the GML. "
                f"Skipping."
            )
            continue

        try:
            lk = build_xy_lookup(cid, unmatched, dwg_room_vertices[cid])
            lookup.update(lk)
            n_shared = len(gml_v) - len(unmatched)
            print(f"  [{cid}] {cell_type}: mapped {len(lk)} room corners "
                  f"({n_shared} shared vertices already in lookup)")
        except AssertionError as exc:
            print(f"  ERROR: {exc}")

    return lookup

# --- RING REPLACEMENT ---
def replace_xy_in_ring(ring_el, lookup, tol=0.05):
    for pos_el in ring_el.findall("gml:pos", NS):
        x, y, z = parse_pos(pos_el.text)
        k = nearest_key(lookup, x, y, tol)
        if k:
            nx, ny = lookup[k]
            pos_el.text = fmt_pos(nx, ny, z)


def replace_boundary_ring_from_lookup(ring_el, global_lookup, tol=0.08):
    """
    Update a boundary LinearRing using the global XY lookup.
    Midpoints between known vertices are linearly interpolated.
    This is for the CellSpaceBoundary midpoint for the doors.
    Vertical planes have their XY replaced and Z preserved.
    """
    positions = ring_el.findall("gml:pos", NS)
    pts       = [parse_pos(p.text) for p in positions]

    floor_idx = [i for i, (x, y, z) in enumerate(pts) if abs(z - FLOOR_Z) < 0.1]
    new_xy: dict[int, tuple[float, float]] = {}

    for i in floor_idx:
        x, y, _ = pts[i]
        k = nearest_key(global_lookup, x, y, tol)
        if k:
            new_xy[i] = global_lookup[k] 

    found = sorted(new_xy.keys())
    for i in floor_idx:
        if i in new_xy:
            continue
        before = [j for j in found if j < i]
        after  = [j for j in found if j > i]
        if before and after:
            j0, j1 = before[-1], after[0]
            ox0, oy0, _ = pts[j0]
            ox1, oy1, _ = pts[j1]
            ox,  oy,  _ = pts[i]
            d_total = np.hypot(ox1 - ox0, oy1 - oy0)
            d_to_i  = np.hypot(ox  - ox0, oy  - oy0)
            frac    = (d_to_i / d_total) if d_total > 1e-9 else 0.5
            nx0, ny0 = new_xy[j0]
            nx1, ny1 = new_xy[j1]
            new_xy[i] = (nx0 + frac * (nx1 - nx0), ny0 + frac * (ny1 - ny0))
        elif before:
            new_xy[i] = new_xy[before[-1]]
        elif after:
            new_xy[i] = new_xy[after[0]]

    old_to_new: dict[tuple[float, float], tuple[float, float]] = {}
    for i in floor_idx:
        if i in new_xy:
            x, y, _ = pts[i]
            old_to_new[(round(x, 4), round(y, 4))] = new_xy[i]

    for i, pos_el in enumerate(positions):
        x, y, z = pts[i]
        k = (round(x, 4), round(y, 4))
        if k in old_to_new:
            nx, ny = old_to_new[k]
            pos_el.text = fmt_pos(nx, ny, z)

# --- UPDATE STATE AND TRANSITION ---

def recompute_states(root) -> dict[str, tuple[float, float, float]]:
    """Recalculate the State point as the centroid of the updated floor ring."""
    state_positions: dict[str, tuple[float, float, float]] = {}
    room_centroids:  dict[str, tuple[float, float]] = {}

    for cs in find_all_cells(root):
        cid = cs.get(f"{{{GML}}}id")
        for ring in cs.findall(".//gml:LinearRing", NS):
            if is_floor_ring(ring):
                verts = ring_xy(ring)
                if verts and verts[0] == verts[-1]:
                    verts = verts[:-1]
                if verts:
                    room_centroids[cid] = centroid(verts)
                break

    for state in root.findall(".//core:State", NS):
        sid  = state.get(f"{{{GML}}}id")
        dual = state.find("core:duality", NS)
        if dual is None:
            continue
        cid = dual.get(f"{{{NS['xlink']}}}href", "").lstrip("#")
        if cid not in room_centroids:
            continue
        cx, cy = room_centroids[cid]
        pos_el = state.find(".//gml:Point/gml:pos", NS)
        if pos_el is not None:
            _, _, z = parse_pos(pos_el.text)
            pos_el.text = fmt_pos(cx, cy, z)
            state_positions[sid] = (cx, cy, z)
            print(f"  State '{sid}': {cid}: ({cx:.4f}, {cy:.4f})")

    return state_positions


def boundary_midpoint(root) -> dict[str, tuple[float, float]]:
    """Returns midpoint of CellSpaceBoundary, which is used in the 3-Point transition LineString in recompute_transitions."""
    
    centroids: dict[str, tuple[float, float]] = {}
    for csb in find_all_boundaries(root):
        bid  = csb.get(f"{{{GML}}}id", "")
        ring = csb.find(".//gml:LinearRing", NS)
        if ring is None:
            continue
        seen: set = set()
        sill: list[tuple[float, float]] = []
        for x, y, z in ring_to_pts(ring):
            if abs(z - FLOOR_Z) < 0.1:
                k = (round(x, 4), round(y, 4))
                if k not in seen:
                    seen.add(k)
                    sill.append((x, y))
        if len(sill) >= 2:
            centroids[bid] = sill[len(sill) // 2]
    return centroids


def recompute_transitions(root, state_positions, boundary_centroids):
    """Rebuild each Transition as: from-state → boundary-centroid → to-state."""
    for trans in root.findall(".//core:Transition", NS):
        tid      = trans.get(f"{{{GML}}}id")
        connects = trans.findall("core:connects", NS)
        if len(connects) < 2:
            print(f"  WARNING: Transition '{tid}' has <2 connects, skipping")
            continue
        s1 = connects[0].get(f"{{{NS['xlink']}}}href", "").lstrip("#")
        s2 = connects[1].get(f"{{{NS['xlink']}}}href", "").lstrip("#")
        if s1 not in state_positions or s2 not in state_positions:
            print(f"  WARNING: missing state position for '{tid}', skipping")
            continue
        x1, y1, z1 = state_positions[s1]
        x2, y2, z2 = state_positions[s2]
        dual_el = trans.find("core:duality", NS)
        dual_id = (dual_el.get(f"{{{NS['xlink']}}}href", "").lstrip("#")
                   if dual_el is not None else "")
        waypoints: list[tuple[float, float, float]] = [(x1, y1, z1)]
        if dual_id in boundary_centroids:
            mx, my = boundary_centroids[dual_id]
            waypoints.append((mx, my, FLOOR_Z))
        waypoints.append((x2, y2, z2))
        ls = trans.find(".//gml:LineString", NS)
        if ls is None:
            continue
        for old in ls.findall("gml:pos", NS):
            ls.remove(old)
        for x, y, z in waypoints:
            p = etree.SubElement(ls, f"{{{NS['gml']}}}pos")
            p.set("srsDimension", "3")
            p.text = fmt_pos(x, y, z)
        route = "via boundary" if len(waypoints) == 3 else "direct"
        print(f"  Transition '{tid}'  {s1} to {s2}  [{route}  dual={dual_id or 'none'}]")

# --- BOUNDS UPDATE ---

def update_envelope(root):
    coords = [parse_pos(p.text) for p in root.findall(".//gml:pos", NS) if p.text]
    if not coords:
        return
    lower = root.find(".//gml:lowerCorner", NS)
    upper = root.find(".//gml:upperCorner", NS)
    if lower:
        lower.text = (f"{min(c[0] for c in coords)} "
                      f"{min(c[1] for c in coords)} "
                      f"{min(c[2] for c in coords)}")
    if upper:
        upper.text = (f"{max(c[0] for c in coords)} "
                      f"{max(c[1] for c in coords)} "
                      f"{max(c[2] for c in coords)}")

# --- MAIN ---

def replace_coordinates(input_path, output_path):
    dwg_room_vertices = load_dwg_vertices(DATA_JSON)

    tree = etree.parse(input_path)
    root = tree.getroot()

    print()
    global_lookup = build_global_lookup(root, dwg_room_vertices)

    print("\nApplying coordinate replacements to all cell geometry …")
    for cs in find_all_cells(root):
        for ring in cs.findall(".//gml:LinearRing", NS):
            replace_xy_in_ring(ring, global_lookup)

    print("\nUpdating boundary geometry …")
    for csb in find_all_boundaries(root):
        bid  = csb.get(f"{{{GML}}}id", "")
        ring = csb.find(".//gml:LinearRing", NS)
        if ring is not None:
            replace_boundary_ring_from_lookup(ring, global_lookup)
            pts  = ring_to_pts(ring)
            seen: set = set()
            sill = []
            for x, y, z in pts:
                if abs(z - FLOOR_Z) < 0.1:
                    k = (round(x, 3), round(y, 3))
                    if k not in seen:
                        seen.add(k); sill.append((round(x, 3), round(y, 3)))
            print(f"  Boundary '{bid}': sill = {sill}")

    print("\nRecomputing State positions …")
    state_positions = recompute_states(root)

    boundary_centroids = boundary_midpoint(root)

    print("\nRecomputing Transition routes …")
    recompute_transitions(root, state_positions, boundary_centroids)

    update_envelope(root)
    tree.write(output_path, xml_declaration=True, encoding="UTF-8", pretty_print=True)
    print(f"\nDone, GML file output to: '{output_path}'")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else INPUT_GML
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_GML
    replace_coordinates(inp, out)
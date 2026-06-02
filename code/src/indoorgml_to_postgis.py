import sys
from lxml import etree
import psycopg2
from pathlib import Path

DIRECTORY  = Path(__file__).resolve().parent.parent.parent
INPUT_GML = DIRECTORY / "data" / "aligned-bk.gml"

# --- DATABASE CONNECTION ---

DB_PARAMS = {
    "host":   "localhost",
    "port":   5432,
    "dbname": "topology",
    "user":   "postgres",
    # Password omitted — uses ~/.pgpass
}

SRID = 0   # local coordinate system

# --- XML NAMESPACES ---

NS = {
    "gml":   "http://www.opengis.net/gml/3.2",
    "core":  "http://www.opengis.net/indoorgml/1.0/core",
    "navi":  "http://www.opengis.net/indoorgml/1.0/navigation",
    "xlink": "http://www.w3.org/1999/xlink",
}
_GML  = NS["gml"]
_CORE = NS["core"]
_NAVI = NS["navi"]

# All CellSpace subtypes used in the new IndoorGML model.
# The parser searches each tag so plain core:CellSpace files still work too.
CELL_SPACE_TAGS = [
    f"{{{_NAVI}}}GeneralSpace",
    f"{{{_NAVI}}}TransitionSpace",
    f"{{{_NAVI}}}ConnectionSpace",
    f"{{{_CORE}}}CellSpace",          # fallback for plain IndoorGML
]

BOUNDARY_TAGS = [
    f"{{{_NAVI}}}ConnectionBoundary",
    f"{{{_CORE}}}CellSpaceBoundary",  # fallback
]

# --- SCHEMA ---

SQL_COMMANDS = """
CREATE EXTENSION IF NOT EXISTS postgis;

DROP TABLE IF EXISTS transition CASCADE;
DROP TABLE IF EXISTS state      CASCADE;
DROP TABLE IF EXISTS room       CASCADE;

CREATE TABLE room (
    id          TEXT PRIMARY KEY,
    room_name   TEXT,
    cell_type   TEXT,
    geom        geometry(PolygonZ, {srid}),
    height      FLOAT,
    file_path   TEXT,
    transform   JSONB
);

CREATE TABLE state (
    id      TEXT PRIMARY KEY,
    room_id TEXT REFERENCES room(id),
    geom    geometry(PointZ, {srid})
);

CREATE TABLE transition (
    id            TEXT PRIMARY KEY,
    from_state    TEXT REFERENCES state(id),
    to_state      TEXT REFERENCES state(id),
    geom          geometry(LineStringZ, {srid}),
    dual_boundary TEXT,
    door_height   FLOAT DEFAULT 3.0
);

CREATE INDEX room_geom_idx       ON room       USING GIST (geom);
CREATE INDEX state_geom_idx      ON state      USING GIST (geom);
CREATE INDEX transition_geom_idx ON transition USING GIST (geom);
""".format(srid=SRID)

# --- GEOMETRY FUNCTIONS ---

def parse_pos(text: str) -> tuple[float, float, float]:
    parts = list(map(float, text.strip().split()))
    return (parts[0], parts[1], parts[2] if len(parts) > 2 else 0.0)

def ring_to_pts(ring_el) -> list[tuple[float, float, float]]:
    return [parse_pos(p.text)
            for p in ring_el.findall("gml:pos", NS)
            if p.text and p.text.strip()]

def wkt_polygonz(pts: list[tuple]) -> str:
    coords = ", ".join(f"{x} {y} {z}" for x, y, z in pts)
    return f"POLYGON Z(({coords}))"

def wkt_pointz(x: float, y: float, z: float) -> str:
    return f"POINT Z({x} {y} {z})"

def wkt_linestringz(pts: list[tuple]) -> str:
    coords = ", ".join(f"{x} {y} {z}" for x, y, z in pts)
    return f"LINESTRING Z({coords})"

def floor_ring_of_solid(solid_el) -> tuple[list[tuple] | None, float]:
    """
    Returns (floor_pts, height_m).
    floor_pts : points from the LinearRing whose Z values are all ~0.
    height_m  : mean Z of the ceiling ring (the ring whose Z values are all > 0).
    """
    floor_pts   = None
    ceiling_zs: list[float] = []
    for ring in solid_el.findall(".//gml:LinearRing", NS):
        pts = ring_to_pts(ring)
        if not pts:
            continue
        zs = [z for _, _, z in pts]
        if all(abs(z) < 0.01 for z in zs):
            floor_pts = pts
        elif all(z > 0.01 for z in zs):
            ceiling_zs.extend(zs)
    height_m = float(sum(ceiling_zs) / len(ceiling_zs)) if ceiling_zs else 0.0
    return floor_pts, height_m

def boundary_sill_line(boundary_el) -> list[tuple] | None:
    """Return the unique floor-level (Z~0) sill points of a boundary ring."""
    ring = boundary_el.find(".//gml:LinearRing", NS)
    if ring is None:
        return None
    pts  = ring_to_pts(ring)
    seen: set = set()
    sill: list[tuple] = []
    for x, y, z in pts:
        if abs(z) < 0.01:
            key = (round(x, 6), round(y, 6))
            if key not in seen:
                seen.add(key)
                sill.append((x, y, 0.0))
    return sill if len(sill) >= 2 else None

# --- PARSE GML ---

def get_cell_type(element) -> str:
    """Return the local tag name, e.g. 'GeneralSpace', 'ConnectionSpace'."""
    tag = element.tag
    return tag.split("}")[1] if "}" in tag else tag

def find_all(root, tags: list[str]):
    elements = []
    for tag in tags:
        elements.extend(root.findall(f".//{tag}"))
    return elements

def parse_gml(path: str) -> dict:
    tree = etree.parse(path)
    root = tree.getroot()

    rooms:       list[dict] = []
    states:      list[dict] = []
    transitions: list[dict] = []

    for cs in find_all(root, CELL_SPACE_TAGS):
        cid       = cs.get(f"{{{_GML}}}id")
        cell_type = get_cell_type(cs)

        name_el   = cs.find("gml:name", NS)
        room_name = name_el.text.strip() if (name_el is not None and name_el.text) else cid

        solid = cs.find(".//gml:Solid", NS)
        floor_pts, height_m = floor_ring_of_solid(solid) if solid is not None else (None, 0.0)
        if floor_pts is None:
            print(f"  WARNING: no floor ring found for '{cid}' ({cell_type})")
            continue
        geom_wkt = wkt_polygonz(floor_pts)

        rooms.append({
            "id":        cid,
            "room_name": room_name,
            "cell_type": cell_type,
            "geom":      geom_wkt,
            "height":    height_m,
            "file_path": None if cell_type == "ConnectionSpace" else f"static/splats/{room_name}.ply",
            "transform": None,
        })
        print(f"  Room '{cid}' ({room_name})  type={cell_type}  "
              f"floor-verts={len(floor_pts)-1}  height={height_m:.2f}m")
        
    boundary_sills: dict[str, str] = {}
    for csb in find_all(root, BOUNDARY_TAGS):
        bid  = csb.get(f"{{{_GML}}}id")
        sill = boundary_sill_line(csb)
        if sill:
            boundary_sills[bid] = wkt_linestringz(sill)
            print(f"  Boundary '{bid}'  sill pts: {len(sill)}")
        else:
            print(f"  WARNING: no sill for boundary '{bid}'")

    for state_el in root.findall(".//core:State", NS):
        sid  = state_el.get(f"{{{_GML}}}id")
        dual = state_el.find("core:duality", NS)
        room_id = None
        if dual is not None:
            room_id = dual.get(f"{{{NS['xlink']}}}href", "").lstrip("#")

        pos_el = state_el.find(".//gml:Point/gml:pos", NS)
        if pos_el is None or not pos_el.text:
            print(f"  WARNING: no geometry for State '{sid}'")
            continue
        x, y, z = parse_pos(pos_el.text)

        states.append({
            "id":      sid,
            "room_id": room_id,
            "geom":    wkt_pointz(x, y, z),
        })
        print(f"  State '{sid}'  room='{room_id}'  pos=({x:.3f},{y:.3f},{z:.3f})")

    for trans_el in root.findall(".//core:Transition", NS):
        tid      = trans_el.get(f"{{{_GML}}}id")
        connects = trans_el.findall("core:connects", NS)
        if len(connects) < 2:
            print(f"  WARNING: Transition '{tid}' has <2 connects, skipping")
            continue

        from_state = connects[0].get(f"{{{NS['xlink']}}}href", "").lstrip("#")
        to_state   = connects[1].get(f"{{{NS['xlink']}}}href", "").lstrip("#")

        dual_el  = trans_el.find("core:duality", NS)
        dual_bid = None
        geom     = None
        if dual_el is not None:
            dual_bid = dual_el.get(f"{{{NS['xlink']}}}href", "").lstrip("#")
            geom     = boundary_sills.get(dual_bid)

        if geom is None:
            print(f"  WARNING: no geometry for Transition '{tid}'")
            continue

        transitions.append({
            "id":            tid,
            "from_state":    from_state,
            "to_state":      to_state,
            "geom":          geom,
            "dual_boundary": dual_bid,
            "door_height":   3.0,
        })
        print(f"  Transition '{tid}'  {from_state} -> {to_state}  (boundary: {dual_bid})")

    return {
        "rooms":       rooms,
        "states":      states,
        "transitions": transitions,
    }

# --- DATABASE FUNCTIONS ---

def run_sql(cur, sql: str):
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)

def insert_rooms(cur, rooms: list[dict]):
    print(f"\nInserting {len(rooms)} room(s) ...")
    for r in rooms:
        cur.execute("""
            INSERT INTO room (id, room_name, cell_type, geom, height, file_path, transform)
            VALUES (
                %(id)s, %(room_name)s, %(cell_type)s,
                ST_GeomFromText(%(geom)s, %(srid)s),
                %(height)s, %(file_path)s, %(transform)s
            )
            ON CONFLICT (id) DO UPDATE SET
                room_name = EXCLUDED.room_name,
                cell_type = EXCLUDED.cell_type,
                geom      = EXCLUDED.geom,
                height    = EXCLUDED.height,
                file_path = EXCLUDED.file_path,
                transform = EXCLUDED.transform
        """, {**r, "srid": SRID})
    print(f"  {len(rooms)} rooms inserted")

def insert_states(cur, states: list[dict]):
    print(f"\nInserting {len(states)} state(s) ...")
    for s in states:
        cur.execute("""
            INSERT INTO state (id, room_id, geom)
            VALUES (%(id)s, %(room_id)s, ST_GeomFromText(%(geom)s, %(srid)s))
            ON CONFLICT (id) DO UPDATE SET
                room_id = EXCLUDED.room_id,
                geom    = EXCLUDED.geom
        """, {**s, "srid": SRID})
    print(f"  {len(states)} states inserted")

def insert_transitions(cur, transitions: list[dict]):
    print(f"\nInserting {len(transitions)} transition(s) ...")
    for t in transitions:
        cur.execute("""
            INSERT INTO transition (id, from_state, to_state, geom, dual_boundary, door_height)
            VALUES (
                %(id)s, %(from_state)s, %(to_state)s,
                ST_GeomFromText(%(geom)s, %(srid)s),
                %(dual_boundary)s, %(door_height)s
            )
            ON CONFLICT (id) DO UPDATE SET
                from_state    = EXCLUDED.from_state,
                to_state      = EXCLUDED.to_state,
                geom          = EXCLUDED.geom,
                dual_boundary = EXCLUDED.dual_boundary,
                door_height   = EXCLUDED.door_height
        """, {**t, "srid": SRID})
    print(f"  {len(transitions)} transitions inserted")

# --- MAIN ---

def main(gml_path: str):
    print(f"Parsing {gml_path} ...\n")
    data = parse_gml(gml_path)

    print(f"\nParsed:  {len(data['rooms'])} rooms  "
          f"{len(data['states'])} states  "
          f"{len(data['transitions'])} transitions")

    print(f"\nConnecting to PostgreSQL at "
          f"{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']} ...")

    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            print("Creating schema ...")
            run_sql(cur, SQL_COMMANDS)
            insert_rooms(cur, data["rooms"])
            insert_states(cur, data["states"])
            insert_transitions(cur, data["transitions"])
        conn.commit()
        print("\nDone.")
    except Exception as e:
        conn.rollback()
        print(f"\nError — rolled back:\n  {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    gml_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_GML
    main(gml_file)
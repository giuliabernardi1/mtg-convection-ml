"""Find which FCI chunks (horizontal strips of the full disc) intersect a region of interest."""
from pathlib import Path

from shapely import wkt
from shapely.geometry import Polygon, box


def load_chunk_footprints(wkt_path: Path) -> dict[str, Polygon]:
    """Parse lines of the form '<chunk_id>,<WKT polygon>'."""
    chunk_footprints = {}
    for line in wkt_path.read_text().splitlines():
        if line.strip():
            chunk_id, polygon_wkt = line.strip().split(",", 1)
            chunk_footprints[chunk_id] = wkt.loads(polygon_wkt)
    return chunk_footprints


def find_relevant_chunks(wkt_path: Path, lon_min, lat_min, lon_max, lat_max) -> list[str]:
    region_of_interest = box(lon_min, lat_min, lon_max, lat_max)
    return sorted(
        chunk_id
        for chunk_id, footprint in load_chunk_footprints(wkt_path).items()
        if region_of_interest.intersects(footprint)
    )
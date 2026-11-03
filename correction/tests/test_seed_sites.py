"""
Tests for seed_sites.py on a synthetic parcel store (no HPC data needed).

    python correction/tests/test_seed_sites.py

Needs duckdb with the spatial extension, geopandas, shapely, h3.
"""
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import config as C
import seed_sites as SS

LAT0, LON0 = 40.0, -100.0
DLAT, DLON = 0.001, 0.0013          # ~111 m squares at 40N


def cell(i, j, w=0.8, h=0.8):
    """A parcel in grid slot (i east, j north), leaving a ~20 m gap to the next slot."""
    x0, y0 = LON0 + i * DLON, LAT0 + j * DLAT
    return box(x0, y0, x0 + w * DLON, y0 + h * DLAT)


def centre(i, j):
    return LAT0 + (j + 0.4) * DLAT, LON0 + (i + 0.4) * DLON


def write_store(root: Path):
    rows = [
        # uuid, geometry, owner
        ("P1", cell(0, 0), "CITY OF XVILLE"),
        ("P2", cell(1, 0), "City of Xville."),    # 0.2 slot gap (~26 m), same owner -> merges with P1
        ("P3", cell(0, 20), "ACME LLC"),           # ~2.2 km north, own site
        ("P4", cell(1, 20), "SOMEONE ELSE"),       # next to P3, different owner -> separate parcel
        ("BIG", box(LON0 + 50 * DLON, LAT0, LON0 + 50 * DLON + 0.05, LAT0 + 0.05), "TOWN"),   # ~25 km2
    ]
    g = gpd.GeoDataFrame({"ll_uuid": [r[0] for r in rows], "owner": [r[2] for r in rows]},
                         geometry=[r[1] for r in rows], crs=4326)
    g["h3_index_9"] = [h3.latlng_to_cell(p.centroid.y, p.centroid.x, 9) for p in g.geometry]
    g["state"] = "ZZ"
    g["wkb_geometry"] = g.geometry.apply(lambda x: x.wkb)
    d = root / "state=ZZ"
    d.mkdir(parents=True)
    pd.DataFrame(g.drop(columns="geometry")).to_parquet(d / "all.parquet", index=False)


def seed(sid, source, origin, lat, lon, independent=True, cid="A"):
    return dict(seed_id=sid, CWNS_ID=cid, STATE_CODE="ZZ", source=source, origin=origin, lat=lat, lon=lon,
                independent=independent, permit_number=None, detail="")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "parcels"
        write_store(root)
        C.PARCEL_BASE = root
        con = C.duckdb_connect()

        seeds = pd.DataFrame([
            seed("A_cwns_point_1", "cwns_point", "cwns", *centre(0, 0)),
            seed("A_icis_fac_frs_1", "icis_fac_frs", "icis_facility", *centre(1, 0)),      # on P2, merges into P1's site
            seed("A_cwns_addr_1", "cwns_addr", "cwns", *centre(0, 0)),                      # same parcel as cwns_point
            seed("A_icis_feat_INF_1", "icis_feat_INF", "icis_feature", centre(0, 20)[0], LON0 - 0.0001),  # just off P3's west edge
            seed("A_icis_fac_addr_1", "icis_fac_addr", "icis_facility", LAT0 + 0.5, LON0 + 0.5, independent=False),  # nowhere
        ])
        m = SS.match_seeds_to_parcels(con, "ZZ", seeds)
        mk = m.set_index("seed_id")
        assert mk.at["A_cwns_point_1", "ll_uuid"] == "P1" and mk.at["A_cwns_point_1", "match_kind"] == "inside"
        assert mk.at["A_icis_fac_frs_1", "ll_uuid"] == "P2"
        assert mk.at["A_icis_feat_INF_1", "ll_uuid"] == "P3" and mk.at["A_icis_feat_INF_1", "match_kind"] == "snapped", mk
        assert 0 < mk.at["A_icis_feat_INF_1", "snap_m"] <= C.SEED_SNAP_M
        assert mk.at["A_icis_fac_addr_1", "match_kind"] == "none"

        sp = seeds.merge(m, on="seed_id")
        parcels = SS.fetch_parcels(con, "ZZ", [u for u in sp["ll_uuid"] if isinstance(u, str)])
        assert parcels.crs.to_epsg() == 5070 and parcels.at["P1", "owner_norm"] == "city of xville"

        sp2, sites = SS.build_sites(sp, parcels)
        assert len(sites) == 2, sites[["site_id", "ll_uuids"]]
        s1 = sites[sites.site_rank == 1].iloc[0]
        assert s1.ll_uuids == ["P1", "P2"] and s1.rep_ll_uuid == "P1"        # CWNS point's parcel is the rep
        assert s1.n_seeds == 3 and s1.n_origins == 2 and s1.has_cwns_point and s1.has_icis_fac_frs
        assert s1.sources == "cwns_addr+cwns_point+icis_fac_frs" and s1.is_best_site and s1.n_sites == 2
        s2 = sites[sites.site_rank == 2].iloc[0]
        assert s2.ll_uuids == ["P3"] and s2.n_origins == 1 and s2.snapped and not s2.has_cwns_point
        assert s1.n_unparcelled_seeds == 1 and s1.plant_n_origins == 3
        assert 2000 < s2.d_to_cwns_point_m < 2500 and abs(s2.d_to_cwns_point_m - s2.d_nearest_other_site_m) < 400
        assert sp2.set_index("seed_id").at["A_icis_feat_INF_1", "site_id"] == "A_s2"
        assert pd.isna(sp2.set_index("seed_id").at["A_icis_fac_addr_1", "site_id"])

        # a copy of the CWNS point adds no origin: only the cwns origin is independent on that site
        copy = pd.DataFrame([
            seed("B_cwns_point_1", "cwns_point", "cwns", *centre(0, 0), cid="B"),
            seed("B_icis_fac_frs_1", "icis_fac_frs", "icis_facility", *centre(0, 0), independent=False, cid="B"),
        ])
        cm = copy.merge(SS.match_seeds_to_parcels(con, "ZZ", copy), on="seed_id")
        _, bsites = SS.build_sites(cm, parcels)
        assert len(bsites) == 1 and bsites.iloc[0].n_origins == 1 and bsites.iloc[0].n_origins_all == 2

        # oversized parcels never merge with a neighbour even with the same owner
        pair = SS.fetch_parcels(con, "ZZ", ["P1", "P2"])
        assert SS.merge_parcels(["P1", "P2"], pair) == {"P1": 0, "P2": 0}              # same owner, ~26 m apart
        pair.loc["P2", "area_m2"] = C.MAX_PARCEL_AREA_M2 + 1                           # now a whole-town artifact
        assert SS.merge_parcels(["P1", "P2"], pair) == {"P1": 0, "P2": 1}
        assert SS.merge_parcels(["P3", "P4"], SS.fetch_parcels(con, "ZZ", ["P3", "P4"])) == {"P3": 0, "P4": 1}  # owners differ


        # A seed can sit on the WRONG parcel, so neighbouring parcels with different owners stay separate sites however
        # close their seeds are (decided 2026-11-03): Stage 1 judges each, and the one holding the truth is the answer.
        la, lo = centre(0, 20)
        close = pd.DataFrame([
            seed("M_cwns_point_1", "cwns_point", "cwns", la, lo, cid="M"),                                  # on P3 (ACME LLC)
            seed("M_icis_fac_frs_1", "icis_fac_frs", "icis_facility", la, LON0 + 1.05 * DLON, cid="M"),     # ~72 m away, on P4 (another owner)
        ])
        tm = close.merge(SS.match_seeds_to_parcels(con, "ZZ", close), on="seed_id")
        assert list(tm.ll_uuid) == ["P3", "P4"]
        _, tsites = SS.build_sites(tm, SS.fetch_parcels(con, "ZZ", ["P3", "P4"]))
        assert len(tsites) == 2 and sorted(tsites.n_origins) == [1, 1], tsites[["site_id", "ll_uuids", "n_origins"]]

        # 01f end to end on the same store: files land per state and read back
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_sites_01f", Path(C.__file__).parent / "01f_build_seed_sites.py")
        f01 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(f01)
        C.SEED_SITES_DIR = Path(tmp) / "seed_sites"
        plants_state = pd.DataFrame({"CWNS_ID": ["A", "B"], "STATE_CODE": ["ZZ", "ZZ"]})
        f01.process_state(con, "ZZ", plants_state, pd.concat([seeds, copy], ignore_index=True))
        out_sites = pd.read_parquet(C.SEED_SITES_DIR / "state=ZZ" / "seed_sites.parquet")
        out_sp = pd.read_parquet(C.SEED_SITES_DIR / "state=ZZ" / "seed_parcels.parquet")
        assert sorted(out_sites.site_id) == ["A_s1", "A_s2", "B_s1"] and len(out_sp) == 7
        print("seed_sites tests passed")


if __name__ == "__main__":
    main()

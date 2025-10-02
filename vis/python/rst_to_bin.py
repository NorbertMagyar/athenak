#!/usr/bin/env python3
"""Convert AthenaK restart dumps (.rst) into standard binary dumps (.bin).

This utility understands the restart layout written by ``RestartOutput`` for
single-file restarts without additional step-3 payloads (no Z4c tracker or
stochastic turbulence state) and reconstructs a conventional binary output that
can be fed to ``vis/python/make_athdf.py`` or the lower-level helpers in
``bin_convert``. By default it targets MHD restarts and emits conserved
variables plus cell-centred magnetic fields, trimming away ghost zones to match
regular dumps.

Limitations
-----------
* Currently assumes the restart does not include Z4c, ADM, radiation, forcing,
  or turbulence metadata (step 3 payload). It throws with a clear error if it
  detects extra bytes between the mesh metadata and the cell data payload.
* Focused on MHD datasets (no separate hydro block in the restart). Hooking in
  additional physics modules mainly requires extending ``_split_block_payload``.
* Requires ``numpy`` (and ``h5py`` only when ``--athdf`` is requested).
"""

from __future__ import annotations

import argparse
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:  # optional dependency for direct athdf export
    from . import bin_convert  # type: ignore
except Exception:  # pragma: no cover - import is best effort for CLI usage
    bin_convert = None  # type: ignore


@dataclass
class RegionIndcs:
    ng: int
    nx1: int
    nx2: int
    nx3: int
    is_: int
    ie: int
    js: int
    je: int
    ks: int
    ke: int
    cnx1: int
    cnx2: int
    cnx3: int
    cis: int
    cie: int
    cjs: int
    cje: int
    cks: int
    cke: int


@dataclass
class RegionSize:
    x1min: float
    x2min: float
    x3min: float
    x1max: float
    x2max: float
    x3max: float
    dx1: float
    dx2: float
    dx3: float


@dataclass
class RestartMeshBlock:
    gid: int
    indices: np.ndarray  # shape (6,), dtype int32
    logical: np.ndarray  # shape (4,), dtype int32
    geometry: np.ndarray  # shape (6,), dtype float64
    data: np.ndarray  # shape (nvars, nz, ny, nx), dtype float64


@dataclass
class RestartContents:
    header_bytes: bytes
    header_lines: List[str]
    time: float
    cycle: int
    mesh_size: RegionSize
    mesh_indcs: RegionIndcs
    mb_indcs: RegionIndcs
    root_level: int
    logical_locations: List[Tuple[int, int, int, int]]
    blocks: List[RestartMeshBlock]
    var_names: List[str]


def _read_parameter_section(fh) -> Tuple[bytes, List[str]]:
    buffer = bytearray()
    marker = b"<par_end>"
    while True:
        chunk = fh.read(4096)
        if not chunk:
            raise ValueError("`<par_end>` marker not found in restart header")
        buffer.extend(chunk)
        idx = buffer.find(marker)
        if idx != -1:
            header_len = idx + len(marker)
            fh.seek(header_len, os.SEEK_SET)
            # consume trailing newline if present
            next_byte = fh.read(1)
            if next_byte != b"\n":
                fh.seek(-1, os.SEEK_CUR)
            raw = bytes(buffer[:header_len])
            lines = [ln.split(b"#")[0].decode("utf-8", "ignore").strip()
                     for ln in raw.splitlines()]
            return raw, [ln for ln in lines if ln]


def _read_region_indices(fh, endian: str = "<") -> RegionIndcs:
    # RegionIndcs contains 19 integers (ng, nx1-3, is/ie/js/je/ks/ke,
    # cnx1-3, cis/cie/cjs/cje/cks/cke).
    fmt = f"{endian}19i"
    values = struct.unpack(fmt, fh.read(struct.calcsize(fmt)))
    return RegionIndcs(*values)


def _read_region_size(fh, real_size: int, endian: str = "<") -> RegionSize:
    fmt = f"{endian}{'f' if real_size == 4 else 'd'}"
    values = [struct.unpack(fmt, fh.read(real_size))[0] for _ in range(9)]
    return RegionSize(*values)


def _compute_block_geometry(mesh: RegionSize, mesh_indcs: RegionIndcs,
                             mb_indcs: RegionIndcs, loc: Tuple[int, int, int, int],
                             root_level: int) -> np.ndarray:
    lx1, lx2, lx3, level = loc
    level_offset = level - root_level
    refinement = 1 << level_offset if level_offset >= 0 else 1

    def _axis(min_val: float, max_val: float, mesh_n: int, mb_n: int, logical: int) -> Tuple[float, float]:
        root_blocks = mesh_n // mb_n
        blocks_at_level = root_blocks * refinement
        block_size = (max_val - min_val) / blocks_at_level
        start = min_val + logical * block_size
        return start, start + block_size

    x1min, x1max = _axis(mesh.x1min, mesh.x1max, mesh_indcs.nx1, mb_indcs.nx1, lx1)
    x2min, x2max = _axis(mesh.x2min, mesh.x2max, mesh_indcs.nx2 or 1, mb_indcs.nx2 or 1, lx2)
    x3min, x3max = _axis(mesh.x3min, mesh.x3max, mesh_indcs.nx3 or 1, mb_indcs.nx3 or 1, lx3)
    return np.array([x1min, x1max, x2min, x2max, x3min, x3max], dtype=np.float64)


def _split_block_payload(payload: memoryview, *, real_size: int, nmhd: int,
                         dims: Tuple[int, int, int], face_dims: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    nx1, nx2, nx3 = dims
    fx1, fx2, fx3 = face_dims
    cells = nx1 * nx2 * nx3
    reals_per_block = cells * nmhd
    real_dtype = np.float32 if real_size == 4 else np.float64

    cons = np.frombuffer(payload[:reals_per_block * real_size], dtype=real_dtype).copy()
    cons = cons.reshape((nmhd, nx3, nx2, nx1))
    offset = reals_per_block * real_size

    def _take(count: int) -> np.ndarray:
        nonlocal offset
        span = count * real_size
        chunk = np.frombuffer(payload[offset:offset + span], dtype=real_dtype).copy()
        offset += span
        return chunk

    bx = _take(fx1).reshape((nx3, nx2, nx1 + 1))
    by = _take(fx2).reshape((nx3, nx2 + 1, nx1))
    bz = _take(fx3).reshape((nx3 + 1, nx2, nx1))
    bcc = np.empty((3, nx3, nx2, nx1), dtype=real_dtype)
    bcc[0] = 0.5 * (bx[:, :, :-1] + bx[:, :, 1:])
    bcc[1] = 0.5 * (by[:, :-1, :] + by[:, 1:, :])
    bcc[2] = 0.5 * (bz[:-1, :, :] + bz[1:, :, :])
    return cons.astype(np.float64), bcc.astype(np.float64)


def _trim_active(data: np.ndarray, mb_indcs: RegionIndcs) -> np.ndarray:
    slicer = (slice(None), slice(mb_indcs.ks, mb_indcs.ke + 1),
              slice(mb_indcs.js, mb_indcs.je + 1),
              slice(mb_indcs.is_, mb_indcs.ie + 1))
    return data[slicer]


def _derive_var_names(nmhd: int) -> List[str]:
    base = ["dens", "mom1", "mom2", "mom3", "ener"]
    names = base[:min(len(base), nmhd)]
    if nmhd > len(base):
        for extra in range(nmhd - len(base)):
            names.append(f"s{extra:02d}")
    names.extend(["bcc1", "bcc2", "bcc3"])
    return names


def _build_meshblocks(chunks: Iterable[bytes], *, real_size: int, nmhd: int,
                      mb_indcs: RegionIndcs, geometry: Sequence[np.ndarray],
                      logical_locations: Sequence[Tuple[int, int, int, int]],
                      root_level: int) -> Tuple[List[RestartMeshBlock], List[str]]:
    nout1 = mb_indcs.nx1 + 2 * mb_indcs.ng
    nout2 = (mb_indcs.nx2 + 2 * mb_indcs.ng) if mb_indcs.nx2 > 1 else 1
    nout3 = (mb_indcs.nx3 + 2 * mb_indcs.ng) if mb_indcs.nx3 > 1 else 1
    face_x = (nout3 * nout2 * (nout1 + 1))
    face_y = (nout3 * (nout2 + 1) * nout1)
    face_z = ((nout3 + 1) * nout2 * nout1)

    blocks: List[RestartMeshBlock] = []
    var_names = _derive_var_names(nmhd)

    for gid, raw in enumerate(chunks):
        view = memoryview(raw)
        cons, bcc = _split_block_payload(view, real_size=real_size, nmhd=nmhd,
                                         dims=(nout1, nout2, nout3),
                                         face_dims=(face_x, face_y, face_z))
        cons = _trim_active(cons, mb_indcs)
        bcc = _trim_active(bcc, mb_indcs)

        stacked = np.concatenate([cons, bcc], axis=0)
        indices = np.array([
            mb_indcs.is_, mb_indcs.ie,
            mb_indcs.js, mb_indcs.je,
            mb_indcs.ks, mb_indcs.ke,
        ], dtype=np.int32)
        loc = logical_locations[gid]
        logical = np.array([loc[0], loc[1], loc[2], loc[3] - root_level], dtype=np.int32)
        blocks.append(RestartMeshBlock(
            gid=gid,
            indices=indices,
            logical=logical,
            geometry=geometry[gid],
            data=stacked,
        ))

    return blocks, var_names


def _parse_restart(path: Path, *, real_size: int, debug: bool = False) -> RestartContents:
    if real_size not in (4, 8):
        raise ValueError("Real size must be 4 or 8 bytes")
    endian = "<"
    filesize = path.stat().st_size

    with path.open("rb") as fh:
        header_bytes, header_lines = _read_parameter_section(fh)
        nmb_total = struct.unpack(f"{endian}i", fh.read(4))[0]
        root_level = struct.unpack(f"{endian}i", fh.read(4))[0]
        mesh_size = _read_region_size(fh, real_size, endian)
        mesh_indcs = _read_region_indices(fh, endian)
        mb_indcs = _read_region_indices(fh, endian)
        time = struct.unpack(f"{endian}{'f' if real_size == 4 else 'd'}", fh.read(real_size))[0]
        dt = struct.unpack(f"{endian}{'f' if real_size == 4 else 'd'}", fh.read(real_size))[0]
        _ = dt  # presently unused but retained for completeness
        cycle = struct.unpack(f"{endian}i", fh.read(4))[0]

        logical_locations = [struct.unpack(f"{endian}4i", fh.read(16))
                              for _ in range(nmb_total)]
        fh.read(4 * nmb_total)  # skip cost_eachmb float32 array

        remainder = fh.read()

    mv = memoryview(remainder)
    if debug:
        print(f"remainder_bytes={len(mv)} nmb_total={nmb_total}")
    data_size = None
    payload_offset = None
    max_step3 = min(len(mv) - 8, 1 << 24)  # allow up to ~16 MiB of metadata
    step_stride = 4 if real_size == 4 else 8
    for step3 in range(0, max_step3 + 1, step_stride):
        remaining = len(mv) - step3 - 8
        if remaining <= 0:
            break
        if remaining % nmb_total != 0:
            continue
        candidate = struct.unpack_from(f"{endian}Q", mv, step3)[0]
        if debug and step3 < 1024:
            print(f"step3={step3} candidate={candidate} remaining={remaining}")
        if candidate > 0 and candidate * nmb_total == remaining:
            data_size = candidate
            payload_offset = step3
            break

    if data_size is None or payload_offset is None:
        raise ValueError(
            "Unable to locate data payload in restart file; try setting --real-size 4 "
            "if this build used single precision."
        )

    step3_bytes = mv[:payload_offset]
    if step3_bytes:
        # Currently unused, but we keep the slice in case further processing is needed.
        step3_bytes.tobytes()

    payload = mv[payload_offset + 8:]
    chunks = [payload[i * data_size:(i + 1) * data_size].tobytes()
              for i in range(nmb_total)]

    nout1 = mb_indcs.nx1 + 2 * mb_indcs.ng
    nout2 = (mb_indcs.nx2 + 2 * mb_indcs.ng) if mb_indcs.nx2 > 1 else 1
    nout3 = (mb_indcs.nx3 + 2 * mb_indcs.ng) if mb_indcs.nx3 > 1 else 1
    face_x = (nout3 * nout2 * (nout1 + 1))
    face_y = (nout3 * (nout2 + 1) * nout1)
    face_z = ((nout3 + 1) * nout2 * nout1)
    total_reals = data_size // real_size
    nmhd = (total_reals - (face_x + face_y + face_z)) // (nout1 * nout2 * nout3)
    if nmhd <= 0:
        raise ValueError("Unable to infer number of MHD variables from restart payload")

    geometries = [
        _compute_block_geometry(mesh_size, mesh_indcs, mb_indcs, loc, root_level)
        for loc in logical_locations
    ]

    blocks, var_names = _build_meshblocks(
        chunks, real_size=real_size, nmhd=nmhd,
        mb_indcs=mb_indcs, geometry=geometries,
        logical_locations=logical_locations, root_level=root_level,
    )
    return RestartContents(
        header_bytes=header_bytes,
        header_lines=header_lines,
        time=time,
        cycle=cycle,
        mesh_size=mesh_size,
        mesh_indcs=mesh_indcs,
        mb_indcs=mb_indcs,
        root_level=root_level,
        logical_locations=list(logical_locations),
        blocks=blocks,
        var_names=var_names,
    )


def _write_bin(path: Path, contents: RestartContents) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nvars = len(contents.var_names)
    data_shape = contents.blocks[0].data.shape
    nz, ny, nx = data_shape[1], data_shape[2], data_shape[3]
    var_line = "  variables:  " + "  ".join(contents.var_names) + "  \n"

    header = (
        "Athena binary output version=1.1\n"
        "  size of preheader=5\n"
        f"  time={contents.time}\n"
        f"  cycle={contents.cycle}\n"
        "  size of location=8\n"
        "  size of variable=4\n"
        f"  number of variables={nvars}\n"
        f"{var_line}"
        f"  header offset={len(contents.header_bytes)}\n"
    )

    with path.open("wb") as fh:
        fh.write(header.encode("utf-8"))
        fh.write(contents.header_bytes)
        for block in contents.blocks:
            fh.write(struct.pack("<6i", *block.indices.tolist()))
            fh.write(struct.pack("<4i", *block.logical.tolist()))
            fh.write(struct.pack("<6d", *block.geometry.tolist()))
            for arr in block.data:
                fh.write(arr.astype(np.float32, copy=False).ravel(order="C").tobytes())


def _build_filedata(contents: RestartContents) -> Dict[str, np.ndarray]:
    mb = contents.blocks
    mb_index = np.stack([blk.indices for blk in mb]).astype(np.int64)
    mb_logical = np.stack([blk.logical for blk in mb]).astype(np.int32)
    mb_geometry = np.stack([blk.geometry for blk in mb]).astype(np.float64)
    mb_data = {
        name: np.stack([blk.data[idx] for blk in mb]).astype(np.float64)
        for idx, name in enumerate(contents.var_names)
    }

    info = {
        "header": contents.header_lines,
        "time": contents.time,
        "cycle": contents.cycle,
        "var_names": contents.var_names,
        "Nx1": contents.mesh_indcs.nx1,
        "Nx2": contents.mesh_indcs.nx2,
        "Nx3": contents.mesh_indcs.nx3,
        "nvars": len(contents.var_names),
        "n_mbs": len(mb),
        "nx1_mb": contents.mb_indcs.nx1,
        "nx2_mb": contents.mb_indcs.nx2,
        "nx3_mb": contents.mb_indcs.nx3,
        "nx1_out_mb": contents.mb_indcs.nx1,
        "nx2_out_mb": contents.mb_indcs.nx2,
        "nx3_out_mb": contents.mb_indcs.nx3,
        "mb_index": mb_index,
        "mb_logical": mb_logical,
        "mb_geometry": mb_geometry,
        "mb_data": mb_data,
    }
    return info


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart", help="path to .rst file produced by AthenaK")
    parser.add_argument("--bin", dest="out_bin", help="output .bin path (defaults to <input>.from_rst.bin)")
    parser.add_argument("--athdf", dest="out_athdf", help="optional athdf output path (requires h5py)")
    parser.add_argument("--real-size", dest="real_size", type=int, default=8,
                        help="size in bytes of AthenaK Real (default: 8)")
    parser.add_argument("--debug", action="store_true",
                        help="print additional diagnostics while parsing")
    args = parser.parse_args(argv)

    restart_path = Path(args.restart).expanduser().resolve()
    if not restart_path.exists():
        raise SystemExit(f"restart file not found: {restart_path}")

    contents = _parse_restart(restart_path, real_size=args.real_size, debug=args.debug)

    out_bin = Path(args.out_bin) if args.out_bin else restart_path.with_suffix(restart_path.suffix + ".from_rst.bin")
    _write_bin(out_bin, contents)
    print(f"wrote binary dump: {out_bin}")

    if args.out_athdf:
        if bin_convert is None:
            raise SystemExit("bin_convert module (and numpy/h5py) required for --athdf output")
        filedata = _build_filedata(contents)
        bin_convert.write_athdf(args.out_athdf, filedata)
        print(f"wrote athdf: {args.out_athdf}")


if __name__ == "__main__":
    main()

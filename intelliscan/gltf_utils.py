import threading
import time
from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np

from utils import log

try:
    import pyvista as pv
except ImportError:
    pv = None

# Lock for 3D model generation to prevent PyVista concurrency issues
model_generation_lock = threading.Lock()


def _export_volume_to_gltf(
    data: np.ndarray,
    gltf_path: Path,
    *,
    origin: np.ndarray | tuple[float, float, float] | None = None,
    spacing: np.ndarray | tuple[float, float, float] | None = None,
    downsample_factor: int = 1,
) -> None:
    """Export a labeled 3D volume to GLTF.

    This helper accepts an in-memory segmentation volume so callers do not need
    to persist an intermediate per-bbox NIfTI file just for GLTF generation.
    """
    if pv is None:
        raise ImportError("PyVista not installed")

    data = np.asarray(data)
    if downsample_factor > 1:
        data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]

    dims = np.array(data.shape)
    origin_arr = np.asarray(origin if origin is not None else (0.0, 0.0, 0.0), dtype=np.float32)
    spacing_arr = np.asarray(spacing if spacing is not None else (1.0, 1.0, 1.0), dtype=np.float32)
    if downsample_factor > 1:
        spacing_arr = spacing_arr * downsample_factor

    unique_classes = np.unique(np.round(data[data > 0.1])).astype(int)
    color_map = {1: "red", 2: "green", 3: "blue", 4: "yellow"}

    with model_generation_lock:
        pl = pv.Plotter(off_screen=True)
        has_mesh = False
        for cls in unique_classes:
            try:
                class_mask = (np.round(data) == cls).astype(float)
                grid = pv.ImageData(dimensions=dims, origin=origin_arr, spacing=spacing_arr)
                grid.point_data["values"] = class_mask.flatten(order="F")
                isosurface = grid.contour([0.5], scalars="values")
                if isosurface.n_points > 0:
                    if downsample_factor > 1 and isosurface.n_points > 10000:
                        isosurface = isosurface.decimate(0.5)
                    color = color_map.get(int(cls), "white")
                    pl.add_mesh(isosurface, color=color, opacity=1.0, smooth_shading=True)
                    has_mesh = True
            except Exception as e:
                log(f"Error processing class {cls}: {e}", level="warning")

        if not has_mesh:
            raise ValueError("No meshes generated for volume")
        gltf_path.parent.mkdir(parents=True, exist_ok=True)
        pl.export_gltf(str(gltf_path))


def generate_gltf_for_sample(sample_id: str, output_base_dir: Path):
    """Helper to generate GLTF model from NIfTI."""
    sample_dir = output_base_dir / sample_id
    nifti_path = sample_dir / "segmentation.nii.gz"
    gltf_path = sample_dir / "model.gltf"

    if not nifti_path.exists():
        raise FileNotFoundError("Segmentation file not found")

    # Check cache
    if gltf_path.exists() and gltf_path.stat().st_mtime > nifti_path.stat().st_mtime:
        return

    if pv is None:
        raise ImportError("PyVista not installed")

    nifti_img = None
    last_exception = None
    data = None
    for _ in range(5):
        try:
            nifti_img = nib.load(nifti_path)
            data = nifti_img.get_fdata()
            break
        except (EOFError, OSError) as e:
            last_exception = e
            time.sleep(1)

    if data is None or nifti_img is None:
        log(f"Failed to read NIfTI file after multiple retries: {last_exception}", level="error")
        raise last_exception

    affine = nifti_img.affine
    origin = affine[:3, 3]
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

    if gltf_path.exists() and gltf_path.stat().st_mtime > nifti_path.stat().st_mtime:
        return

    _export_volume_to_gltf(
        data,
        gltf_path,
        origin=origin,
        spacing=spacing,
        downsample_factor=3,
    )


def generate_bump_gltf(sample_id: str, bump_id: str, output_base_dir: Path) -> str:
    """Helper to generate GLTF model for a single bump."""
    sample_dir = output_base_dir / sample_id
    nifti_path = sample_dir / "mmt" / "pred" / f"pred_{bump_id}.nii.gz"
    gltf_path = sample_dir / "mmt" / "pred" / f"pred_{bump_id}.gltf"

    if not nifti_path.exists():
        raise FileNotFoundError(f"Bump segmentation not found: {nifti_path}")

    if gltf_path.exists() and gltf_path.stat().st_mtime > nifti_path.stat().st_mtime:
        return f"/output/{sample_id}/mmt/pred/pred_{bump_id}.gltf"

    if pv is None:
        raise ImportError("PyVista not installed")

    nifti_img, last_exception, data = None, None, None
    for _ in range(5):
        try:
            nifti_img = nib.load(nifti_path)
            data = nifti_img.get_fdata()
            break
        except (EOFError, OSError) as e:
            last_exception = e
            time.sleep(0.5)

    if data is None:
        raise IOError(f"Could not read bump file: {last_exception}")

    affine = nifti_img.affine
    origin = affine[:3, 3]
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

    try:
        if gltf_path.exists() and gltf_path.stat().st_mtime > nifti_path.stat().st_mtime:
            return f"/output/{sample_id}/mmt/pred/pred_{bump_id}.gltf"
        _export_volume_to_gltf(data, gltf_path, origin=origin, spacing=spacing)
    except Exception as e:
        log(f"Error generating bump GLTF {bump_id}: {e}", level="error")
        raise

    return f"/output/{sample_id}/mmt/pred/pred_{bump_id}.gltf"


def generate_all_bump_gltfs(sample_id: str, output_base_dir: Path) -> int:
    """Generate GLTF models for all bumps in the sample."""
    sample_dir = output_base_dir / sample_id
    pred_dir = sample_dir / "mmt" / "pred"
    if not pred_dir.exists():
        return 0

    files = list(pred_dir.glob("pred_*.nii.gz"))
    log(f"Generating GLTF models for {len(files)} bumps in {sample_id}...")

    for f in files:
        try:
            bump_id = f.name.replace("pred_", "").replace(".nii.gz", "")
            generate_bump_gltf(sample_id, bump_id, output_base_dir)
        except Exception as e:
            log(f"Error generating bump GLTF for {f.name}: {e}", level="warning")

    return len(files)


def generate_all_bump_gltfs_from_results(
    sample_id: str,
    results: Iterable[object],
    output_base_dir: Path,
) -> int:
    """Generate per-bbox GLTF files directly from in-memory segmentation results."""
    sample_dir = output_base_dir / sample_id
    seg_reference = sample_dir / "segmentation.nii.gz"
    generated = 0

    for result in results:
        bump_id = int(getattr(result, "bbox_index"))
        prediction = np.asarray(getattr(result, "prediction"))
        gltf_path = sample_dir / "mmt" / "pred" / f"pred_{bump_id}.gltf"

        try:
            if seg_reference.exists() and gltf_path.exists() and gltf_path.stat().st_mtime > seg_reference.stat().st_mtime:
                generated += 1
                continue
            _export_volume_to_gltf(prediction, gltf_path)
            generated += 1
        except Exception as e:
            log(f"Error generating bump GLTF for pred_{bump_id}: {e}", level="warning")

    return generated

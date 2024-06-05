from pathlib import Path
from datetime import datetime
import h5py
import rasterio
from rasterio.transform import from_origin
from typing import Tuple, List, Callable
import os
import xarray as xr


def process_h5_filename(file_path: Path) -> Tuple[str, str]:
    filename = file_path.name

    date_str = filename.split(".")[1][1:]
    date = datetime.strptime(date_str, "%Y%j").date()
    date_str = date.strftime("%Y-%m-%d")

    return date_str, "0"


def list_datasets(h5_path: Path):
    with h5py.File(h5_path, "r") as h5_file:

        def print_name(name):
            print(name)

        h5_file.visit(print_name)


def get_metadata(h5_file: h5py.File) -> dict:
    metadata = {}
    if "HDFEOS INFORMATION/StructMetadata.0" in h5_file:
        struct_metadata = h5_file["HDFEOS INFORMATION/StructMetadata.0"][()].decode()
        for line in struct_metadata.split("\n"):
            if "UpperLeftPointMtrs" in line:
                ul = line.split("=")[1].strip("()").split(",")
                metadata["ul_x"], metadata["ul_y"] = float(ul[0]), float(ul[1])
            if "LowerRightMtrs" in line:
                lr = line.split("=")[1].strip("()").split(",")
                metadata["lr_x"], metadata["lr_y"] = float(lr[0]), float(lr[1])

    if (
        "ul_x" not in metadata
        or "ul_y" not in metadata
        or "lr_x" not in metadata
        or "lr_y" not in metadata
    ):
        print("Geospatial metadata not found in HDFEOS INFORMATION/StructMetadata.0")
        return {}

    return metadata


def convert_h5_to_tiff(
    h5_path: Path, dataset_name: str, output_tiff_path: Path
) -> None:
    with h5py.File(h5_path, "r") as h5_file:
        grid_path = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
        dataset_full_name = f"{grid_path}/{dataset_name}"

        # Check if the dataset exists
        if dataset_full_name not in h5_file:
            print(f"Dataset '{dataset_full_name}' not found in file: {h5_path}")
            list_datasets(h5_path)
            return

        dataset = h5_file[dataset_full_name]

        data = dataset[:]
        metadata = get_metadata(h5_file)

        if not metadata:
            print(
                f"Missing metadata for dataset '{dataset_full_name}' in file: {h5_path}"
            )
            return

        ul_x, ul_y, lr_x, lr_y = (
            metadata["ul_x"],
            metadata["ul_y"],
            metadata["lr_x"],
            metadata["lr_y"],
        )
        nrows, ncols = data.shape
        xres = (lr_x - ul_x) / float(ncols)
        yres = (ul_y - lr_y) / float(nrows)

        transform = from_origin(ul_x, ul_y, xres, yres)

        # Write to TIFF
        try:
            with rasterio.open(
                output_tiff_path,
                "w",
                driver="GTiff",
                height=nrows,
                width=ncols,
                count=1,
                dtype=data.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(data, 1)
                dst.update_tags(**{k: v for k, v in metadata.items() if v is not None})
            print(f"Successfully wrote {output_tiff_path}")
        except Exception as e:
            print(f"Failed to write {output_tiff_path}, Error: {e}")


def process_folder(
    input_folder: Path, dataset_names: List[str], output_folder: Path
) -> None:
    if not input_folder.exists():
        print(f"Input folder does not exist: {input_folder}")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Output folder: {output_folder}")

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".h5"):
                h5_path = Path(root) / file
                print(f"Processing file: {h5_path}")
                date_str, band = process_h5_filename(h5_path)
                for dataset_name in dataset_names:
                    output_tiff_path = (
                        output_folder
                        / f"{h5_path.stem}_{dataset_name.split('/')[-1]}_{date_str}.tif"
                    )
                    try:
                        convert_h5_to_tiff(h5_path, dataset_name, output_tiff_path)
                        print(
                            f"Converted {h5_path} dataset '{dataset_name}' to {output_tiff_path}"
                        )
                    except Exception as e:
                        print(
                            f"Failed to process dataset '{dataset_name}' in file: {h5_path}, Error: {e}"
                        )


# Example usage
input_folder = Path("/Users/yeseongmoon/VIIRS")
dataset_names = [
    "DNB_At_Sensor_Radiance",
    "QF_Cloud_Mask",
    "QF_VIIRS_M10",
    "QF_VIIRS_M11",
    "QF_VIIRS_M12",
    "QF_VIIRS_M13",
    "QF_VIIRS_M15",
    "QF_VIIRS_M16",
]
output_folder = Path("/Users/yeseongmoon/TIF")

process_folder(input_folder, dataset_names, output_folder)


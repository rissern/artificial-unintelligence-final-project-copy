from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
import rasterio

@dataclass
class BoundingBox:
    north_bound_latitude: float
    south_bound_latitude: float
    east_bound_longitude: float
    west_bound_longitude: float

def process_jp2_filename(file_path: Path) -> tuple[str, str]:
    """
    Process the filename of a JP2 file and return the date and band.

    Example filename: T34PCA_20240605T085601_AOT_60m.jp2
    """
    filename = file_path.name

    # Extract the date from the filename
    date_str = filename.split("_")[1]
    date = date_str.split("T")[0]

    # Extract the band from the filename
    band = filename.split("_")[2]

    return date, band

def get_all_data_files(base_path: Path) -> list[Path]:
    """
    List the datasets in a JP2 file.
    """

    return list(base_path.rglob("T34PCA_*_B*_60m.jp2"))

def get_data_xml_file(base_path: Path) -> Path:
    """
    Find the XML file associated with a JP2 file.
    """
    return base_path / "INSPIRE.xml"
    

def read_xml_file(xml_path: Path) -> BoundingBox:
    """
    Read an XML file and return the contents as a dictionary.
    """
    with open(xml_path, "r") as xml_file:
        parsed_xml = ElementTree.parse(xml_file)
    
    # find coordinates in xml file
    # under the tag gmd:EX_GeographicBoundingBox

    # find the tag gmd:EX_GeographicBoundingBox
    geographic_bounding_box = parsed_xml.find(".//{http://www.isotc211.org/2005/gmd}EX_GeographicBoundingBox")

    # find the tag gmd:northBoundLatitude
    north_bound_latitude = geographic_bounding_box.find(".//{http://www.isotc211.org/2005/gmd}northBoundLatitude")
    north_bound_latitude = north_bound_latitude.find(".//{http://www.isotc211.org/2005/gco}Decimal").text
    
    # find the tag gmd:southBoundLatitude
    south_bound_latitude = geographic_bounding_box.find(".//{http://www.isotc211.org/2005/gmd}southBoundLatitude")
    south_bound_latitude = south_bound_latitude.find(".//{http://www.isotc211.org/2005/gco}Decimal").text

    # find the tag gmd:eastBoundLongitude
    east_bound_longitude = geographic_bounding_box.find(".//{http://www.isotc211.org/2005/gmd}eastBoundLongitude")
    east_bound_longitude = east_bound_longitude.find(".//{http://www.isotc211.org/2005/gco}Decimal").text

    # find the tag gmd:westBoundLongitude
    west_bound_longitude = geographic_bounding_box.find(".//{http://www.isotc211.org/2005/gmd}westBoundLongitude")
    west_bound_longitude = west_bound_longitude.find(".//{http://www.isotc211.org/2005/gco}Decimal").text

    return BoundingBox(
        north_bound_latitude=float(north_bound_latitude),
        south_bound_latitude=float(south_bound_latitude),
        east_bound_longitude=float(east_bound_longitude),
        west_bound_longitude=float(west_bound_longitude)
    )


def convert_jp2_to_tiff(jp2_path: Path, out_path: Path, bounding_box: BoundingBox) -> None:
    """
    Convert a JP2 file to a TIFF file.
    """

    # Read the JP2 file
    with rasterio.open(jp2_path) as src:
        profile = src.profile
        profile.update(
            driver="GTiff",
            count=1,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=0,
        )

        # Write the TIFF file
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(src.read(1), 1)

            # Add the bounding box to the TIFF file
            dst.update_tags(
                north_bound_latitude=bounding_box.north_bound_latitude,
                south_bound_latitude=bounding_box.south_bound_latitude,
                east_bound_longitude=bounding_box.east_bound_longitude,
                west_bound_longitude=bounding_box.west_bound_longitude,
            )

            print(f"Converted JP2 to TIFF: {out_path}")       


def process_raw_sentinet2_frame(input_folder: Path, output_folder: Path) -> None:
    """
    Process a folder containing Sentinel-2 data.
    """

    # create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find all JP2 files
    jp2_files = get_all_data_files(input_folder)
    xml_file = get_data_xml_file(input_folder)
    bounding_box = read_xml_file(xml_file)

    for jp2_file in jp2_files:
        date, band = process_jp2_filename(jp2_file)
        output_tiff_path = output_folder / f"L2A_{date}_{band}.tif"
        convert_jp2_to_tiff(jp2_file, output_tiff_path, bounding_box)


if __name__ == '__main__':
    BASE_PATH = Path("/Users/joshcordero/Downloads/S2A_MSIL2A_20240605T085601_N0510_R007_T34PCA_20240605T151501.SAFE")

    process_raw_sentinet2_frame(BASE_PATH, BASE_PATH / "output")
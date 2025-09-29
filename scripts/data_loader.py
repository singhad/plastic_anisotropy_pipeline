import pandas as pd
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

def read_dataset(file_path: Path) -> pd.DataFrame:
    """
    Reads a CSV file from the specified path and returns it as a pandas DataFrame.

    Args:
        file_path (Path): Path object pointing to the CSV file to be read.

    Returns:
        pd.DataFrame: DataFrame containing the dataset read from the CSV file.
    """
    if not file_path.exists():
        logger.error(f"The file {file_path} does not exist.")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    logger.info(f"Reading dataset from: {file_path}")
    return pd.read_csv(file_path)

def extract_property_grid(df: pd.DataFrame, property_name: str) -> pd.DataFrame:
    """
    Extracts Time, Temperature, and a specified property from the dataset for interpolation.

    Args:
        df (pd.DataFrame): Cleaned dataset.
        property_name (str): Column name of the property to interpolate (e.g. 'Rp0.2').

    Returns:
        pd.DataFrame: A DataFrame with columns [Time, Temperature, Property]
    """
    required_columns = ["Time", "Temperature", property_name]
    if not all(col in df.columns for col in required_columns):
        logger.error(f"One or more required columns not found: {required_columns}")
        raise ValueError(f"One or more required columns not found: {required_columns}")

    logger.info(f"Extracting grid for property: {property_name}")
    return df[required_columns].dropna()

def list_available_properties(df: pd.DataFrame) -> List[str]:
    """
    Identifies numerical properties in the dataset suitable for interpolation.

    Args:
        df (pd.DataFrame): Dataset containing all columns.

    Returns:
        List[str]: Names of numeric columns excluding metadata.
    """
    exclude_columns = {"Time", "Temperature", "Supplier", "Orientation", "Treatment", "Unnamed: 8"}
    properties = [col for col in df.columns if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])]
    logger.info(f"Available numeric properties: {properties}")
    return properties

def clean_data(
    df: pd.DataFrame,
    df_r_values: pd.DataFrame,
    y_strain: float = 1.5
) -> pd.DataFrame:
    """
    Cleans the main dataset by removing invalid rows (from supplier '3D Pro' and Time <= 0).
    Merges the r-values, i.e., r_bar and delta_r (from df_r_values) into the main dataset.

    Args:
        df (pd.DataFrame): Raw dataset containing mechanical properties.
        df_r_values (pd.DataFrame): Dataset containing r-values to merge.
        y_strain (float): Axial Strain (%) value for filtering. Default is 1.5%.

    Returns:
        pd.DataFrame: Filtered and merged dataset for valid experiments.
    """
    cleaned = df.copy()
    if "Supplier" in cleaned.columns:
        cleaned = cleaned[cleaned["Supplier"] != "3D Pro"]
    if "Time" in cleaned.columns:
        cleaned = cleaned[cleaned["Time"] > 0]
        
    # filter to the strain level used in the R script by default
    y_strain_str = "Axial Strain: "+str(y_strain)+ "%"
    df_r_values_copy = df_r_values.copy()
    if "y_strain" in df_r_values_copy.columns and y_strain_str in df_r_values_copy["y_strain"].values:
        df_r_values_copy = df_r_values_copy[df_r_values_copy["y_strain"] == y_strain_str]
    elif "y_strain" in df_r_values_copy.columns and y_strain != 1.5:
        logger.warning(f"No matching strain level '{y_strain_str}' found in r-values. Using default 1.5%.")

    # Renaming columns to match
    df_r_values_copy.rename(columns={"r_bar": "R_bar", "Delta_r": "delta_R"}, inplace=True)

    # keep only what we need
    if "Time" in df_r_values_copy.columns:
        df_r_values_copy = df_r_values_copy[df_r_values_copy["Time"] > 0]
    keep = [c for c in ["Time", "Temperature", "R_bar", "delta_R"] if c in df_r_values_copy.columns]
    df_r_values_copy = df_r_values_copy[keep]

    # merge
    merged = cleaned.merge(df_r_values_copy, on=["Time", "Temperature"], how="left")

    return merged

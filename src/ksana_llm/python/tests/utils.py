# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================


import csv
import logging
import yaml

# Configure logging for utils
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def modify_yaml_field(file_path, field_path_str, new_value):
    """
    Modify or add a field in a YAML file based on a dot-separated field path.

    Args:
        file_path (str): Path to the YAML file.
        field_path_str (str): Dot-separated path to the field.
        new_value: New value to set for the field.
    """
    with open(file_path, "r") as file:
        content = yaml.safe_load(file) or {}

    field_path = field_path_str.split(".")
    temp = content
    for field in field_path[:-1]:
        temp = temp.setdefault(field, {})

    temp[field_path[-1]] = new_value

    with open(file_path, "w") as file:
        yaml.dump(content, file, default_flow_style=False)


def read_from_csv(csv_file, col_idx=0, remove_head=True):
    """
    Read a specific column from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        col_idx (int): Index of the column to read.
        remove_head (bool): Whether to skip the header row.

    Returns:
        List[str]: List of values from the specified column.
    """
    logger.debug(f"Reading CSV file: {csv_file}")
    with open(csv_file, "r", newline="") as f:
        csv_reader = csv.reader(f)
        if remove_head:
            next(csv_reader, None)
        return [row[col_idx] for row in csv_reader]

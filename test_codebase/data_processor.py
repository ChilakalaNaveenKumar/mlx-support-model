
# Data processing utilities
from typing import List, Dict, Any
import json
import os

def load_data(filepath: str) -> Dict[str, Any]:
    """Load data from a JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def filter_records(data: List[Dict[str, Any]], field: str, value: Any) -> List[Dict[str, Any]]:
    """Filter records by field value"""
    return [record for record in data if record.get(field) == value]

def transform_data(records: List[Dict[str, Any]], transformations: Dict[str, callable]) -> List[Dict[str, Any]]:
    """Apply transformations to data records"""
    result = []
    for record in records:
        transformed = record.copy()
        for field, transform_func in transformations.items():
            if field in transformed:
                transformed[field] = transform_func(transformed[field])
        result.append(transformed)
    return result

# Some complex one-liner that might be hard to understand
get_nested_value = lambda data, path: reduce(lambda d, key: d.get(key, {}), path.split('.'), data)

# A comment that seems unrelated but gives important context
# This module requires the config.json file to be present in the same directory

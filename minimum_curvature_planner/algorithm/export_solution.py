import csv
import numpy as np
from perception_data import Centreline

def export_solution(points: np.ndarray, output_path: str, format: str = 'csv'):
    if format == 'csv':
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for point in points:
                writer.writerow(point)
    else:
        raise ValueError(f"Unsupported format: {format}")

import csv
import json
import numpy as np
from perception_data import Centreline

def export_solution(points: np.ndarray, output_path: str, format: str = 'csv'):
    if format == 'csv':
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for point in points:
                writer.writerow(point)
    elif format == 'json':
        json_data = [
            {
                'pose': {
                    'position': {
                        'x': float(points[i, 0]),
                        'y': float(points[i, 1]),
                        'z': 0.0
                    }
                }
            }
            for i in range(points.shape[0])
        ]
        with open(output_path, mode='w') as file:
            json.dump(json_data, file, indent=4)
    else:
        raise ValueError(f"Unsupported format: {format}")

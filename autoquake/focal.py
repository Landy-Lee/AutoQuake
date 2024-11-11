import os
import subprocess
from pathlib import Path


class GAfocal:
    def __init__(self, dout_file_name: str, result_path: Path):
        self.dout_file = dout_file_name
        self.main_dir = Path(__file__).parents[1] / 'GAfocal'
        self.result_path = result_path

    def run(self):
        subprocess.run(
            ['./gafocal'], input=self.dout_file.encode() + b'\n', cwd=self.main_dir
        )
        os.system(
            f"cp {self.main_dir / 'results.txt'} {self.result_path / 'gafocal_catalog.txt'}"
        )

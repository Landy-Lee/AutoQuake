import subprocess
from pathlib import Path


class GAfocal:
    def __init__(self, dout_file_name: str):
        self.dout_file = dout_file_name
        self.main_dir = Path(__file__).parents[1] / 'GAfocal'

    def run(self):
        subprocess.run(
            ['./gafocal'], input=self.dout_file.encode() + b'\n', cwd=self.main_dir
        )

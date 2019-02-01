import subprocess
from glob import glob
import os


def test_examples():
    cwd = os.getcwd()

    os.chdir(os.path.join(cwd, 'examples'))

    for fn in glob('params_tutorial.py'):
        if fn == 'mae.py':
            continue
        subprocess.check_call(['python', fn])

    os.chdir(cwd)

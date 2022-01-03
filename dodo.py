import pathlib

# DOIT_CONFIG = {'default_tasks': []}

# Directory containing ``dodo.py``
WORKING_DIR = pathlib.Path(__file__).parent.resolve()
# Path to ``build`` folder
BUILD_DIR = WORKING_DIR.joinpath('build')
# Dict of subfolders in ``build``
SUBDIRS = {
    dir: BUILD_DIR.joinpath(dir) for dir in [
        'datasets',
        'figures',
        'hydra_outputs',
        'mprof_outputs',
        'cvd_figures',
    ]
}


def task_build_dir():
    """Create ``build`` directory and subdirectories."""

    def make_subdir(subdir):
        subdir.mkdir(parents=True, exist_ok=True)

    for (subdir_name, subdir) in SUBDIRS.items():
        yield {
            'name': subdir_name,
            'actions': [(make_subdir, [subdir])],
            'targets': [subdir],
        }

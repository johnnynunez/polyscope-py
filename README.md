# polyscope-py
Python bindings for Polyscope. https://polyscope.run/py

[![travis build status](https://travis-ci.com/nmwsharp/polyscope-py.svg?branch=master)](https://travis-ci.com/nmwsharp/polyscope-py)
[![appveyor build status](https://ci.appveyor.com/api/projects/status/epf2tpgc0oarjrrx/branch/master?svg=true)](https://ci.appveyor.com/project/nmwsharp/polyscope-py/branch/master)
[![PyPI](https://img.shields.io/pypi/v/polyscope?style=plastic)](https://pypi.org/project/polyscope/)
[![Conda](https://img.shields.io/conda/v/conda-forge/polyscope)](https://anaconda.org/conda-forge/polyscope)

This library is a python wrapper and deployment system. The core library lives at https://github.com/nmwsharp/polyscope. See documentation at https://polyscope.run/py.

### Installation

```
python -m pip install polyscope
```

or

```
conda install -c conda-forge polyscope
```

polyscope-py should work out-of-the-box on any combination of Python 2.7, 3.4-3.8 and Linux/macOS/Windows. Your graphics hardware must support OpenGL >= 3.3 core profile.

## For developers

This repo is configured with CI on both travis and appveyor. Travis is useful for running tests and quick builds, but appveyor generates final deployment wheels, because it supports a broader combination of platforms.

### Deploy a new version

- Commit the desired version to the `master` branch, be sure the version string in `setup.py` corresponds to the new version number.
- Watch the travis & appveyor builds to ensure the test & build stages succeed and all wheels are compiled (takes ~1 hr on Travis, ~2 hrs on Appveyor).
- While you're waiting, update the docs, including the changelog.
- Tag the commit with a tag like `v1.2.3`, matching the version in `setup.py`. This will kick off a new Appveyor build which deploys the wheels to PyPI after compilation.
- Update the conda builds by committing to the [feedstock repository](https://github.com/conda-forge/polyscope-feedstock). This generally just requires bumping the version number and updating the hash in `meta.yml`. Since `meta.yml` is configured to pull source from PyPi, you can't do this until after the source build has been uploaded from Appveyor (generally <15 min).

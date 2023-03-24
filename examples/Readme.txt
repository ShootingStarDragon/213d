for building with pyInstaller:
first make sure you have Poetry so all dependencies are installed by the pyproject.toml file
tutorial is here: https://kivyschool.com/installation/
Once poetry is installed:
    poetry update
    poetry shell
    cd to this folder
    to build backgroundsubtraction.py using the supplied .spec file:
        python -m PyInstaller Backsub.spec --clean
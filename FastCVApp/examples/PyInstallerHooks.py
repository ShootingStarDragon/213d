#!/usr/bin/env python3

#blosc2 hook adapted from numpy hook so blosc2 python can be packaged with pyinstaller, since it's the same team that made numpy (and both use BSD 3-Clause License) I assume all the license stuff is the same --ShootingStarDragon
# --- Copyright Disclaimer ---
#
# In order to support PyInstaller with numpy<1.20.0 this file will be duplicated for a short period inside
# PyInstaller's repository [1]. However this file is the intellectual property of the NumPy team and is
# under the terms and conditions outlined their repository [2].
#
# .. refs:
#
#   [1] PyInstaller: https://github.com/pyinstaller/pyinstaller/
#   [2] NumPy's license: https://github.com/numpy/numpy/blob/master/LICENSE.txt
#   [3] Python-Blosc2's license: https://github.com/Blosc/python-blosc2/blob/main/LICENSE.txt
#
"""
This hook should collect all binary files and any hidden modules that numpy needs.

Our (some-what inadequate) docs for writing PyInstaller hooks are kept here:
https://pyinstaller.readthedocs.io/en/stable/hooks.html

PyInstaller has a lot of NumPy users so we consider maintaining this hook a high priority.
Feel free to @mention either bwoodsend or Legorooj on Github for help keeping it working.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect all DLLs inside numpy's installation folder, dump them into built app's root.
binaries = collect_dynamic_libs("blosc2", ".")

# Submodules PyInstaller cannot detect (probably because they are only imported by extension modules, which PyInstaller
# cannot read).
# hiddenimports = ['numpy.core._dtype_ctypes']
hiddenimports = ['blosc2.blosc2_ext']



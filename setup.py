#!/usr/bin/env python

import os, sys, re

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension


if __name__ == "__main__":
    import sys


    import numpy
    from Cython.Build import cythonize


    # Set up the C++-extension.
    libraries = []
    if os.name == "posix":
        libraries.append("m")
    include_dirs = [
        "pysiggen",
        "mjd_siggen",
        numpy.get_include(),
    ]

    src = [os.path.join("mjd_siggen", fn) for fn in [
        "calc_signal.c",
        "cyl_point.c",
        "detector_geometry.c",
        "fields.c",
        "mjd_fieldgen.c",
        "point.c",
        "read_config.c",
#            "siggen_helpers.c",
    ]]
    
    src += [
        os.path.join("pysiggen", "_pysiggen.pyx"),
    ]

    major, minor1, minor2, release, serial = sys.version_info
    if major >= 3:
        def rd(filename):
            f = open(filename, encoding="utf-8")
            r = f.read()
            f.close()
            return r
    else:
        def rd(filename):
            f = open(filename)
            r = f.read()
            f.close()
            return r

    vre = re.compile("__version__ = \"(.*?)\"")
    m = rd(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "pysiggen", "__init__.py"))
    version = vre.findall(m)[0]

    ext = Extension(
        "pysiggen._pysiggen",
        sources=src,
        language="c",
        libraries=libraries,
        include_dirs=include_dirs,
#            extra_compile_args=["-std=c++11",
#                                "-Wno-unused-function",
#                                "-Wno-uninitialized",
#                                "-DNO_THREADS"],
#            extra_link_args=["-std=c++11"],
    )
    extensions = cythonize([ext])


setup(
    name="pysiggen",
    version=version,
    author="Ben Shanks",
    author_email="benjamin.shanks@gmail.com",
    packages=["pysiggen"],
    ext_modules=extensions,
)

# from setuptools import setup
# from Cython.Build import cythonize


# setup(
#     ext_modules = cythonize("forward_prop_traj.pyx")
# )

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [
    Extension(
        "forward_prop_traj",
        ["forward_prop_traj.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="Forward propagate trajectory",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)

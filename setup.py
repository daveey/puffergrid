from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

def build_ext(srcs, package="puffergrid"):
    return Extension(
        name=package + "." + srcs[0].split('/')[-1].split('.')[0],
        sources=srcs,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        language="c++",
        build_dir='build',
        include_dirs=[numpy.get_include()],
    )

ext_modules = [
    build_ext(["puffergrid/action.pyx"]),
    build_ext(["puffergrid/event.pyx"]),
    build_ext(["puffergrid/grid.cpp"]),
    build_ext(["puffergrid/grid_env.pyx"]),
    build_ext(["puffergrid/grid_object.pyx"]),
    build_ext(["puffergrid/observation_encoder.pyx"]),
    build_ext(["puffergrid/stats_tracker.pyx"]),
    build_ext(["examples/forage.pyx"], "puffergrid.examples"),
]

optimized = True
setup(
    name='puffergrid',
    version='0.1',
    packages=find_packages(),
    ext_modules=cythonize(
        ext_modules,
        language="c++",
        build_dir='build',
        compiler_directives={
            "profile": True,

            "language_level": "3",
            "embedsignature": not optimized,
            "annotation_typing": not optimized,
            "cdivision": not optimized,
            "boundscheck": not optimized,
            "wraparound": not optimized,
            "initializedcheck": not optimized,
            "nonecheck": not optimized,
            "overflowcheck": not optimized,
            "overflowcheck.fold": not optimized,
            "linetrace": not optimized,
            "c_string_encoding": "utf-8",
            "c_string_type": "str",
        },
        annotate=True,
    ),
    description='',
    url='https://github.com/daveey/puffergrid',
    install_requires=[
        'numpy',
        'cython',
        'tqdm',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

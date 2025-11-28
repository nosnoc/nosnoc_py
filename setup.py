from distutils.core import setup

setup(
    name='nosnoc',
    version='0.1',
    python_requires='>=3.8',
    description='Nonsmooth Numerical Optimal Control for Python',
    #    url='',
   author='Anton Pozharskiy, Jonathan Frey, Armin Nurkanovic',
    # use_scm_version={
    #   "fallback_version": "0.1-local",
    #   "root": "../..",
    #   "relative_to": __file__
    # },
   license='BSD',
    packages = ["nosnoc"],
    include_package_data = True,
    py_modules=[],
    setup_requires=['setuptools_scm'],
    install_requires=[
        'numpy>=1.20.0',
        'scipy',
        'casadi',
        'matplotlib',
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'parametrized'
        ]
)

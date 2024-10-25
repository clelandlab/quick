from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
def get_version(rel_path):
    return (here / rel_path).read_text().strip()

setup(
    name="clelandlab-quick",
    version=get_version("quick/VERSION"),
    description="QuICK is a useful wrap of QICK.",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/clelandlab/quick",
    author="Cleland Lab",
    author_email="clelandlab@proton.me",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="QICK,quantum,experiment,measurement,qubit,control,readout,fpga",
    packages=find_packages(where="."),
    python_requires=">=3.6, <4",
    install_requires=[
        "qick==0.2.291",
        "numpy",
        "scipy",
        "pyyaml",
        "pyro4",
        "matplotlib",
        "ipython"
    ],
    package_data={
        "quick": ["VERSION", "constants/*"]
    },
    include_package_data=True,
    project_urls={
        "Source": "https://github.com/clelandlab/quick",
        "Documentation": "https://clelandlab-quick.readthedocs.io/en/latest/",
        "Tracker": "https://github.com/clelandlab/quick/issues",
    },
)

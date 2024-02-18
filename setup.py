from setuptools import find_packages, setup

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="OpenSMDREnv",
    version="0.0.1",
    author="Rohan Thorat, Juhi Singh, Rajdip Nayek",
    author_email="rohanthorat2@gmail.com",
    description="Solve the KdV equation using statFEM",
    long_description=readme,
    url="https://github.com/rohan-v-thorat/OpenSMDREnv",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(exclude=[
        
    ]),
    python_requires=">=3.6",
    install_requires=[
        "numpy", "h5py"
    ]
)

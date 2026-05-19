# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="working-memory-eeg-analysis", 
    version="0.0.1",
    author="Marius Keute and Silvana Miranda Montenegro",
    description="Analysis functions for working-memory behavioral and EEG data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['project_functions'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7'
)

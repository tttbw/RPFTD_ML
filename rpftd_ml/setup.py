from setuptools import setup, find_packages

setup(
    name="rpftd_ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "torch>=1.9.0",
        "akshare>=1.10.0",
        "matplotlib>=3.4.0",
    ],
    author="tttbw",
    author_email="your.email@example.com",
    description="A Financial Data Denoising Method Based on Reflection-Padded Fourier Transform",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tttbw/RPFTD_ML",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 
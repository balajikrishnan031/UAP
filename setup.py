from setuptools import setup, find_packages

setup(
    name="tmrm",
    version="1.0.0",
    description="Topological Manifold Resonant Machine: Ground-up High-Performance Predictive Machine Learning",
    long_description=open("README_TMRM.md", encoding="utf-8").read() if open("README_TMRM.md") else "",
    long_description_content_type="text/markdown",
    author="Antigravity AI Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    entry_points={
        "console_scripts": [
            "tmrm=tmrm.cli:main"
        ]
    }
)

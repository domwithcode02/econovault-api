"""
EconoVault Python SDK Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="econovault",
    version="1.0.0",
    author="EconoVault Team",
    author_email="support@econovault.com",
    description="Python SDK for the EconoVault Economic Data API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/econovault/econovault-python-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "aiohttp>=3.8.0",
        "python-dateutil>=2.8.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=2.10.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.910",
            "flake8>=4.0.0",
        ],
        "pandas": [
            "pandas>=1.3.0",
        ],
        "numpy": [
            "numpy>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "econovault=econovault.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
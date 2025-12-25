"""
CHRONOS: Cryptocurrency High-Risk Observation & Novelty-detection Operational System
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "CHRONOS: Cryptocurrency AML Detection with Temporal GNNs and Counterfactual Explanations"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                # Remove version specifiers for PyTorch packages (handle separately)
                if line.startswith('torch'):
                    continue
                requirements.append(line)
        return requirements

setup(
    name='chronos-aml',
    version='0.1.0',
    description='Cryptocurrency AML Detection with Temporal GNNs and Counterfactual Explanations',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author='CHRONOS Team',
    author_email='',
    url='',
    packages=find_packages(exclude=['tests', 'scripts', 'docs']),
    python_requires='>=3.10,<3.11',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.4.2',
            'pytest-cov>=4.1.0',
            'black>=23.10.0',
            'flake8>=6.1.0',
            'mypy>=1.6.1',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'ipykernel>=6.26.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='cryptocurrency aml graph-neural-networks temporal explainability counterfactual',
    project_urls={
        'Documentation': '',
        'Source': '',
        'Bug Reports': '',
    },
    include_package_data=True,
    zip_safe=False,
)

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="famews",
    version=get_version("famews/__init__.py"),
    author="Marine Hoche, Olga Mineeva, Manuel Burger",
    license="MIT License",
    description="FAMEWS: A Fairness Auditing tool for Medical Early-Warning Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["famews"],
    python_requires=">=3.9",
    install_requires=[
        'pandas',
        'pytorch',
        'numpy',
        'pandas<2.0',
        'scikit-learn>=1.2.0',
        'jupyter',
        'seaborn>=0.12.2',
        'matplotlib>=3.6.2',
        'pytest==7.1.2',
        'mypy==0.981',
        'tqdm>=4.64',
        'pyarrow>=8.0.0',
        'pytables>=3.7.0',
        'black==22.6.0',
        'isort==5.9.3',
        'pyspark>=3.2.1',
        'gin-config>=0.5.0',
        'tensorboard>=2.11.0',
        'coolname>=2.2.0',
        'lightgbm>=3.3.5',
        'plotly>=5.9.0',
        'shap>=0.41.0',
        'setuptools',
        'wheel',
        'twine',
        'pathos>=0.2.9',
        'scikit-fda>=0.8.1',
        'coloredlogs>=15.0',
        'types-PyYAML',
        'reportlab>=4.0.4'
        'rbo>=0.1.3'
    ]
)

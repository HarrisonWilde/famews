import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="famews",
    version="1.0",
    author="-",
    description="FAMEWS: A Fairness Auditing tool for Medical Early-Warning Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["famews"],
    python_requires=">=3.9",
)

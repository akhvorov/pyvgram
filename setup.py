import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyvgram",
    version="0.0.4",
    author="Aleksandr Khvorov",
    author_email="khvorov.aleksandr@gmail.com",
    description="VGram tokenization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akhvorov/pyvgram",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
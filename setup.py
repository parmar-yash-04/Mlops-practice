from setuptools import find_packages, setup

setup(
    name="mlops",
    version="0.0.1",
    author="mlops",
    author_email="mlops@gmail.com",
    packages=find_packages(),
    install_requires=[
        "psycopg2-binary",
        "sqlalchemy",
        "pandas",
        "numpy",
        "scikit-learn",
        "pyyaml",
        "python-dotenv",
        "from_root",
    ],
)

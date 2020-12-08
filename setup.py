# Automatically created by: shub deploy

from setuptools import setup, find_packages

setup(
    name='movielens_crawler',
    version='1.0',
    author='minhdao',
    description='MovieLens Crawler Project',
    packages=find_packages(exclude=[
        'docs',
        'tests',
        'static',
        'templates',
        '.gitignore',
        'README.md',
        'data'
    ]),
    entry_points={'scrapy': ['settings = movielens.settings']},
    install_requires=[
        'scrapy',
        'w3lib',
        'pylint',
        'autopep8',
        'rope',
        'python-dotenv',
        'dnspython',
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn'
    ]
)

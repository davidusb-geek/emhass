"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='emhass',  # Required
    version='0.1.5',  # Required
    description='An Energy Management System for Home Assistant',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/davidusb-geek/emhass',  # Optional
    author='David HERNANDEZ',  # Optional
    author_email='davidusb@gmail.com',  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        "Operating System :: OS Independent",
    ],
    keywords='energy, management, optimization, hass',  # Optional
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.8, <4',
    install_requires=[
        'numpy>=1.20.1',
        'pandas>=1.2.3',
        'pvlib>=0.8.1',
        'protobuf>=3.0.0a3',
        'siphon>=0.9',
        'pytz>=2021.1',
        'requests>=2.25.1',
        'beautifulsoup4>=4.9.3',
        'pulp>=2.4',
        'pyyaml>=5.4.1',
        'netcdf4>=1.5.3',
        'tables>=3.6.1',
    ],  # Optional
    entry_points={  # Optional
        'console_scripts': [
            'emhass=emhass.command_line:main',
        ],
    },
)

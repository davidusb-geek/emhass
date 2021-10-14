"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Command
import pathlib, os, sys

NAME = 'emhass', # Required
VERSION = '0.2.0',  # Required
DESCRIPTION = 'An Energy Management System for Home Assistant', # Optional
URL = 'https://github.com/davidusb-geek/emhass',  # Optional
AUTHOR = 'David HERNANDEZ',  # Optional
EMAIL = 'davidusb@gmail.com',  # Optional

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')

        sys.exit()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name=NAME
    version=VERSION
    description=DESCRIPTION
    url=URL
    author=AUTHOR
    author_email=EMAIL
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
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
        'pvlib>=0.9.0',
        'protobuf>=3.0.0',
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

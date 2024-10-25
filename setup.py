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
    version='0.11.0',  # Required
    description='An Energy Management System for Home Assistant',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/davidusb-geek/emhass',  # Optional
    author='David HERNANDEZ',  # Optional
    author_email='davidusb@gmail.com',  # Optional
    classifiers=[  # Optional
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
        "Operating System :: OS Independent",
    ],
    keywords='energy, management, optimization, hass',  # Optional
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.10, <3.12',
    install_requires=[
        'wheel', 
        'numpy==1.26.4',
        'scipy==1.12.0',
        'pandas<=2.0.3',
        'pvlib>=0.10.2',
        'protobuf>=3.0.0',
        'pytz>=2021.1',
        'requests>=2.25.1',
        'beautifulsoup4>=4.9.3',
        'h5py==3.12.1',
        'pulp>=2.4',
        'pyyaml>=5.4.1',
        'tables<=3.9.1',
        'skforecast==0.13.0',
        'flask>=2.0.3',
        'waitress>=2.1.1',
        'plotly>=5.6.0'
    ],  # Optional
    entry_points={  # Optional
        'console_scripts': [
            'emhass=emhass.command_line:main',
        ],
    },
    package_data={'emhass': ['templates/index.html','templates/template.html','templates/configuration.html','static/advanced.html','static/basic.html', 'static/script.js', 'static/configuration_script.js',
    'static/style.css','static/configuration_list.html','static/img/emhass_icon.png','static/img/emhass_logo_short.svg', 'static/img/feather-sprite.svg','static/data/param_definitions.json',
    'data/cec_modules.pbz2', 'data/cec_inverters.pbz2','data/associations.csv','data/config_defaults.json']},
)

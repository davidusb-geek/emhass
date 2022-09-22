Create a developer environment:
```
virtualenv -p /usr/bin/python3 emhass-dev
```
To develop using Anaconda use (pick the correct Python and Pip versions):
```
conda create --name emhass-dev python=3.8 pip=21.0.1
```
Then activate environment and install the required packages using:
```
pip install -r requirements.txt
```
Add `emhass` to the Python path using the path to `src`, for example:
```
/home/user/emhass/src
```
If working on linux we can add these lines to the `~/.bashrc` file:
```
# Python modules
export PYTHONPATH="${PYTHONPATH}:/home/user/emhass/src"
```
Don't foget to source the `~/.bashrc` file:
```
source ~/.bashrc
```
Update the build package:
```
python3 -m pip install --upgrade build
```
And generate distribution archives with:
```
python3 -m build
```
Or with:
```
python3 setup.py build bdist_wheel
```
Create a new tag version:
```
git tag vX.X.X
git push origin --tags
```
Upload to pypi:
```
twine upload dist/*
```

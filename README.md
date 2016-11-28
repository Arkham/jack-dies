# Jack Dies

An exploration in data science predicting survival rate for Titanic passengers.

## Mac OSX Setup

You want to install python from homebrew

`brew install python`

You want to install `virtualenv` and `virtualenv-wrapper`

`pip install virtualenv && pip install virtualenv-wrapper`

Then create a new virtualenv and activate it

`mkvirtualenv jack-dies && workon jack-dies`

Now you can install all the dependencies from the requirements file

`pip install -r requirements.txt`

Then just run this command to get matplotlib to run

`echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc`

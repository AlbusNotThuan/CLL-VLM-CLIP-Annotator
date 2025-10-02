import os, sys

def get_root_path():
    if "__file__" in globals():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))       # run in .py file
    else:
        return os.getcwd()      # handle case run in notebook
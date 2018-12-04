"""
install networkx version 2.2 through terminal:
python -m pip install --upgrade pip
python -m pip install networkx
"""
import os
import subprocess

if __name__ == "__main__":
    if (True): # todo modify here to switch mode

        dir = "DATA/"
        for filename in os.listdir(dir):
            print filename
            subprocess.call(["python", "code/Main.py", "--inst", dir+filename, "--alg", "LS2", "--time", "600", "--seed", "2935812"], shell=True)
            subprocess.call(["python", "code/Main.py", "--inst", dir+filename, "--alg", "LS2", "--time", "600", "--seed", "2523643"], shell=True)
            subprocess.call(["python", "code/Main.py", "--inst", dir+filename, "--alg", "LS2", "--time", "600", "--seed", "3496034"], shell=True)
# TSP
CSE 6140 Final Project

## Required
For the code to run on Ubuntu 18.04 <br />
```
sudo apt install build-essential
sudo apt install default-jdk
sudo apt install python
sudo apt install python-numpy
sudo apt install python-matplotlib
```

## Data
All original data stored in DATA folder

## Input matrix
Stored as csv file in input folder. (No need to read file when computing, only stored for plotting simplicity)

## Output
### Solution files
File name: `<instance>_<method>_<cutoff>[_<random_seed>].sol` <br />
random_seed is only applicable when the method of choice is local search, i.e. LS1 or LS2 <br />
File format: <br />
* line 1: best quality
* line 2: list of vertex IDs of the TSP tour

### Solution trace files
File name: `<instance>_<method>_<cutoff>[_<random_seed>].trace` <br />
random_seed is only applicable when the method of choice is local search, i.e. LS1 or LS2 <br />
File format: <br />
* Each line contains timestamp in seconds and quality of the best found solution at that point in time

## Run Main function
Make sure in TSP folder. In your terminal, type command as following:
### GNU
```
python code/Main.py --inst <filename> --alg [BnB | Approx | LS1 | LS2] --time <cutoff_in_seconds> [--seed <random_seed>]
```
Example running Boston.tsp with MST approximation algorithm with cutoff time 100 seconds:
```
python code/Main.py --inst DATA/Boston.tsp --alg Approx --time 100
```

### Unix
```
python code/Main.py -i <filename> -a [BnB | Approx | LS1 | LS2] -t <cutoff_in_seconds> [-s <random_seed>]
```
Example running .tsp with local search 1 algorithm with cutoff time 100 seconds with random seed 12345:
```
python code/Main.py --inst DATA/Cincinnati.tsp --alg LS1 --time 100 -seed 12345
```

## Run input sample
Make sure in TSP folder. In your terminal, type command as following:
### GNU
```
python code/Input.py --inst <filename> --alg [BnB | Approx | LS1 | LS2] --time <cutoff_in_seconds> [--seed <random_seed>]
```
### Unix
```
python code/Input.py -i <filename> -a [BnB | Approx | LS1 | LS2] -t <cutoff_in_seconds> [-s <random_seed>]
```

## Run output sample
Make sure in TSP folder. In your terminal, type command as following:
```
python code/Output.py
```

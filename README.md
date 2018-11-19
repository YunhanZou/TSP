# TSP
CSE 6140 Final Project

## Required
Python version: 2.7 <br />
networkx==2.2 <br />
numpy==1.13.3

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
python src/Main.py --inst <filename> --alg [BnB | Approx | LS1 | LS2] --time <cutoff_in_seconds> [--seed <random_seed>]
```
Example running Boston.tsp with MST approximation algorithm with cutoff time 100 seconds:
```
python src/Main.py --inst DATA/Boston.tsp --alg Approx --time 100
```

### Unix
```
python src/Main.py -i <filename> -a [BnB | Approx | LS1 | LS2] -t <cutoff_in_seconds> [-s <random_seed>]
```
Example running .tsp with local search 1 algorithm with cutoff time 100 seconds with random seed 12345:
```
python src/Main.py --inst DATA/Cincinnati.tsp --alg LS1 --time 100 -seed 12345
```

## Run input sample
Make sure in TSP folder. In your terminal, type command as following:
### GNU
```
python src/Input.py --inst <filename> --alg [BnB | Approx | LS1 | LS2] --time <cutoff_in_seconds> [--seed <random_seed>]
```
### Unix
```
python src/Input.py -i <filename> -a [BnB | Approx | LS1 | LS2] -t <cutoff_in_seconds> [-s <random_seed>]
```

## Run output sample
Make sure in TSP folder. In your terminal, type command as following:
```
python src/Output.py
```

## Run Approx
install networkx version 2.2 through terminal:
```
python -m pip install --upgrade pip
python -m pip install networkx
```

Switch mode by modifying src/Approximation.py as follow: <br />
'False' To generate outputs:
(generated in output folder)  <br />
Make sure in TSP folder
```
./run_Approx.sh
```
'True' To generate graphs:
(currently: rel_time vs dim)  <br />
Make sure in TSP folder
```
python src/Approximation.py
```

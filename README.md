# TSP
CSE 6140 Final Project

## Run input
Make sure in TSP folder
### GNU
```
python src/Input.py --inst <filename> --alg [BnB | Approx | LS1 | LS2] --time <cutoff_in_seconds> [--seed <random_seed>]
```
### Unix
```
python src/Input.py -i <filename> -a [BnB | Approx | LS1 | LS2] -t <cutoff_in_seconds> [-s <random_seed>]
```

## Run output sample
Make sure in TSP folder
```
python src/Output.py
```

## Run Approx
install networkx version 2.2 through terminal:
python -m pip install --upgrade pip
python -m pip install networkx

Switch mode by modifying src/Approximation.py as follow:
'Flase' To generate outputs:
(in output folder)
```
./run_Approx.sh
```
'True' To generate graphs:
(currently: rel_time vs dim)
```
python src/Approximation.py
```
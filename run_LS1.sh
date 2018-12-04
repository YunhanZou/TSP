
for filename in DATA/*.tsp; do
    # echo $filename
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 12345
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 35313
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 45631
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 90593
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 2935812
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 2523643
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 3496034
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 425
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 634506
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 540963
done
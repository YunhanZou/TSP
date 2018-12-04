
for filename in DATA/*.tsp; do
    # echo $filename
    python code/Main.py --inst $filename --alg LS2 --time 600 --seed 425
done
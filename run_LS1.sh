
for filename in DATA/*.tsp; do
    # echo $filename
    python code/Main.py --inst $filename --alg LS1 --time 600 --seed 12345
done
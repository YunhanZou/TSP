
for filename in DATA/*.tsp; do
    # echo $filename
    python code/Main.py --inst $filename --alg BnB --time 600
done

for filename in DATA/*.tsp; do
    # echo $filename
    python src/Main.py --inst $filename --alg BnB --time 100
done

for filename in DATA/*.tsp; do
    # echo $filename
    python code/Main.py --inst $filename --alg Approx --time 0
done
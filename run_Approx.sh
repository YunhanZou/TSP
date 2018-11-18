
for filename in DATA/*.tsp; do
    # echo $filename
    python src/Approximation.py -i $filename -a Approx -t 0
done
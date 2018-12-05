n=2
type=QRTD
width=5
	
for filename in DATA/*.tsp; do
    filename="$(cut -d'.' -f1 <<<"$filename")"
    filename="$(cut -d'/' -f2 <<<"$filename")"
    #echo $filename
    echo "\includegraphics[width=${width}cm]{plots/${filename}_LS${n}_${type}.png}"
done

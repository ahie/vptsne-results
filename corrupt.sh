for c in 0.1 0.2 0.3 0.4
do
  for i in $(seq 1 20)
  do
    python3 corrupt.py $c
  done
done

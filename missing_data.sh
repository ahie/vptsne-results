for c in 0.90 0.93 0.96 0.99
do
  for i in $(seq 1 20)
  do
    python3 missing_data.py $c
  done
done

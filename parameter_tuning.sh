mkdir parameter_tuning_output
for batch_size in 200 400
do
  for run_id in $(seq 1 20)
  do
    python3 parameter_tuning.py 30 $batch_size $run_id
  done
done

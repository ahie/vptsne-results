plt_types = ["knn", "trustworthiness"];
plt_titles = ["1-NN", "Trustworthiness (k=12)"];

for j = 1:length(plt_types)
  for batch_size = [200, 400]
    for i = 1:20
      s = "_3_30_" + string(batch_size) + "_" + string(i) + ".log";
      ptsne_data(:,i) = movavg(fscanf(fopen("parameter_tuning_output/" + plt_types(j) + "_ptsne" + s, 'r'), '%f', 30), 'e', 1); 
      vptsne_data(:,i) = movavg(fscanf(fopen("parameter_tuning_output/" + plt_types(j) + "_vptsne" + s, 'r'), '%f', 30), 'e', 1); 
    end 
    subplot(120 + j)
    hold on
    plterr(vptsne_data.', 1:50:1500, "VPTSNE " + batch_size)
    plterr(ptsne_data.', 1:50:1500, "PTSNE " + batch_size)
    title(plt_titles(j))
    legend('Location', 'south', 'Orientation', 'horizontal')
    hold off 
  end 
end

function plterr(data, x, name)
  ci = 0.95;
  alpha = 1-ci;
  n = size(data,1);
  t_multiplier = tinv(1-alpha/2, n-1);
  err = t_multiplier*std(data)/sqrt(n);
  errorbar(x, mean(data), err, 'DisplayName', name)
end


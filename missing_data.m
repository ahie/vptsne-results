plt_types = ["knn_score", "trustworthiness"];
plt_titles = ["1-NN", "Trustworthiness (k=12)"];
num_samples = 20

for j = 1:length(plt_types)
  i = 1;
  for downsampling = ["0.9", "0.93", "0.96", "0.99"]
    "missing_data_output/ptsne_subset_" + plt_types(j) + "_" + downsampling + ".log"
    ptsne_data(:,i) = fscanf(fopen("missing_data_output/ptsne_subset_" + plt_types(j) + "_" + downsampling + ".log", 'r'), '%f', num_samples);
%    The worse, deterministic variant is left out of the plot
%    vptsne_data(:,i) = fscanf(fopen("missing_data_output/vptsne_subset_" + plt_types(j) + "_"  + downsampling + ".log", 'r'), '%f', num_samples);
    vptsne2_data(:,i) = fscanf(fopen("missing_data_output/vptsne2_subset_" + plt_types(j) + "_"  + downsampling + ".log", 'r'), '%f', num_samples);
    i = i + 1;
  end
  subplot(120 + j);
  hold on
  p1 = plterr(vptsne2_data, 90:3:99, "VPTSNE");
  %s1 = scatter(reshape(repmat(90:3:99,num_samples,1), 1, []), reshape(vptsne2_data, 1, []));
  %s1.set('CData', p1.get('Color'));
  p2 = plterr(ptsne_data, 90:3:99, "PTSNE");
  %s2 = scatter(reshape(repmat(90:3:99,num_samples,1), 1, []), reshape(ptsne_data, 1, []));
  %s2.set('CData', p2.get('Color'));
  ylabel(plt_titles(j));
  xlabel("% of training data discarded");
  legend([p1, p2], 'Location', 'south', 'Orientation', 'horizontal');
  % line plot best scores
  %plot(90:3:99, max(vptsne2_data))
  %plot(90:3:99, max(ptsne_data))
  hold off
end

function [s] = plterr(data, x, name)
  ci = 0.95;
  alpha = 1-ci;
  n = size(data,1);
  t_multiplier = tinv(1-alpha/2, n-1);
  err = t_multiplier*std(data)/sqrt(n);
  s = errorbar(x, mean(data), err, 'DisplayName', name);
end


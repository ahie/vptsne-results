plt_types = ["knn_score", "trustworthiness"];
score_types = ["knn", "trustworthiness"];
plt_titles = ["1-NN", "Trustworthiness (k=12)"];
num_datapoints = 20;

for j = 1:length(plt_types)
  i = 1;
  for corruption_level = ["0.1", "0.2", "0.3", "0.4"]
    s = score_types(j) + "_" + corruption_level + ".log";
    ptsne_data(:,i) = fscanf(fopen("corrupted_output/ptsne_" + s, 'r'), '%f', num_datapoints);
    vptsne_data(:,i) = fscanf(fopen("corrupted_output/vptsne_" + s, 'r'), '%f', num_datapoints);
    i = i + 1;
  end
  subplot(120 + j);
  hold on
  p1 = plterr(vptsne_data, 0.1:0.1:0.4, "VPTSNE");
%  s1 = scatter(reshape(repmat(0.1:0.1:0.4, num_datapoints, 1), 1, []), reshape(vptsne_data, 1, []));
%  s1.set('CData', p1.get('Color'));
  p2 = plterr(ptsne_data, 0.1:0.1:0.4, "PTSNE");
%  s2 = scatter(reshape(repmat(0.1:0.1:0.4, num_datapoints, 1), 1, []), reshape(ptsne_data, 1, []));
%  s2.set('CData', p2.get('Color'));
  ylabel(plt_titles(j));
  xlabel("% corrupted");
  legend([p1, p2], 'Location', 'south', 'Orientation', 'horizontal');
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


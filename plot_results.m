function [f_cost,f_grad] = plot_results(labels, times, cost, grad, sty)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

f_cost = figure
for i=1:length(labels)
  semilogy(times{i}-times{i}(1), cost{i}, sty{i}); hold on;
end  
title('Cost function value'); 
legend(labels{:});
hold off;

f_grad = figure
for i=1:length(labels)
  semilogy(times{i}-times{i}(1), grad{i}, sty{i}); hold on;
end
title('Norm of the gradient'); 
legend(labels{:});
hold off;
end


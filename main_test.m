%% Main test reproducing example from the article
clear all;

%% 1. Random example
% How it was generated
% n = 10; L = 5;
% A = rand(n,n,L) + 1i * rand(n,n,L);
% save('rand_mat.mat','A');
load('rand_mat');

%% Optimize with various algorithms
Nsweeps = 100;
[jacobi_labels, jacobi_cost, jacobi_grad, jacobi_times, Uhs] = ...
    run_tests_jacobi(A, Nsweeps);
[manopt_solvers,manopt_labels,manopt_cost, ...
    manopt_grad,manopt_info,manopt_times,manopt_Uh] = run_tests_manopt(A);

%% Plot results
[f1,f2]= plot_results([jacobi_labels; manopt_labels],[jacobi_times;manopt_times], ...
             [jacobi_cost;manopt_cost],[jacobi_grad;manopt_grad], ...
             {'r-'; 'g-'; 'b-';'m-'; 'k-'; 'c-'});
           
%% adjust axes if needed          
figure(f1);xlim([0,1.5]);
figure(f2);xlim([0,1.5]);


%% 2. Diagonal example
% How it was generated
% n = 20; L= 20;
% D = zeros(n,n,L);
% 
% In = eye(n);
% for i=1:L
%   D(:,:,i) = diag(ones(n,1) + In(:,i)); 
% end
% X = randn(n);
% [Ust,~,~] = svd(X);
% A = matr_rotate(D, Ust) + 1e-4*randn(n,n,L);
% save('diag_mat.mat','A');
load('diag_mat.mat');

%% Optimize with various algorithms
Nsweeps = 20;
[jacobi_labels, jacobi_cost, jacobi_grad, jacobi_times, Uhs] = ...
    run_tests_jacobi(A, Nsweeps);
[manopt_solvers,manopt_labels,manopt_cost, ...
    manopt_grad,manopt_info,manopt_times,manopt_Uh] = run_tests_manopt(A);

%% Plot results
[f3,f4] = plot_results([jacobi_labels;manopt_labels],[jacobi_times;manopt_times], ...
             [jacobi_cost;manopt_cost],[jacobi_grad;manopt_grad], ...
             {'ro-'; 'gx-'; 'b+-';'m*-'; 'ks-'; 'cv-'});
%% adjust axes if needed          
figure(f3);xlim([0,0.5]);
figure(f4);xlim([0,0.5]);





    
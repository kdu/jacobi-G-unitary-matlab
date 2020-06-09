function [Uhat, info] = JacobiG_2_max(A, maxiter, batch_size, U0, grad_eps, x_eps)
%JACOBI_G_2_max Run the Jacobi-G-max algorithm for nxnxL approximate diagonalization  
  n = size(A,1);
  d = 2;
  L = size(A,3);

  % Check the parameters
  if (~exist('maxiter', 'var')), maxiter = 9999; end
  if (~exist('grad_threshold')), grad_eps = 1e-16; end
  if (~exist('x_threshold')), x_eps = 1e-16; end
  if (~exist('opt_disp')), opt_disp = 0; end

  disp('Running Jacobi-G-max with parameters:');
  disp(sprintf(['maxiter = %d, grad_eps = %g, x_eps = %g'], ...
        maxiter, grad_eps, x_eps));
  norm_A2 = norm(A(:))^2;
  if (opt_disp ~= 0)
    disp(sprintf('Processing tensor, d = %d, n = %d, ||A||^2 = %f', ...
                 d, n, norm(A(:))^2));  
    disp(sprintf(['   k | f(U_k) - ||A||^2\t| ||ProjGrad|| \t|'...
                  ' (i_k,j_k)\t| c_k      \t| h''(0)       \t|'...
                  'h''(x_k) \t|(f_k-f_{k-1})/(h''(0)|x_k|)']));     
    disp(['-----------------------------------------------------------'...
          '-------------------------------']);
  end
  
  tic
  % Perform the zero-th iteration
  W_k = zeros(n,n,L);  
  if (~exist('U0', 'var') || isempty(U0))
    U_k = eye(n); W_k = A;
  else  
    U_k = U0; W_k = matr_rotate(A, U0, d);
  end
  
  maxsweep = (ceil(maxiter/batch_size));
  iter_pairs = zeros(maxsweep+1, 2);
  iter_progress = zeros(maxsweep+1, 5);
  iter_times =  zeros(maxsweep+1,1);
  
  k = 0;
  f_k = matr_sumdiag2(W_k);
  Lambda_k = Lambda_2(W_k);
  norm_Lambda_k = norm(Lambda_k, 'fro');
  if (opt_disp ~= 0)
    disp(sprintf(['%4d | %10.5e \t| %10.5e \t|         \t|         \t|' ...
                  '         \t|'], k, norm_A2-f_k, norm_Lambda_k));
  end   
  iter_pairs(1,:) = [0,0];
  iter_progress(1,:) = [norm_A2- f_k,norm_Lambda_k, 0, 0, 0];
  iter_times(1) = toc;

  times = 0;
  for k=1:maxiter
    % Choose maximal element in the gradient
    [~,max_ind] = max(abs(Lambda_k(:)));
    i = floor((max_ind-1)/n)+1;
    j = mod((max_ind-1),n)+1;
      
    if (i == j)
      break;
    end
    if (i > j)
      t = j; j = i; i = t;
    end
    
    % Find the elementary rotation
    h0 = sqrt(2)* abs(Lambda_k(i,j));
    [Psi_k,c_k,~,~] = find_Jacobi(W_k, i, j);
    G_k = givens_complex(n, i, j, Psi_k);
          
    % Update the matrices, current iterate, and Riemannian gradient
    W_k = matr_rotate(W_k, G_k);      
    U_k(:, [i,j]) =  U_k(:, [i,j]) * Psi_k;
    Lambda_k = Lambda_2(W_k);
         
    % Save/output results
    if (mod(k, batch_size) == 0 || k == maxiter)
      times = times+1;
      norm_Lambda_k = norm(Lambda_k, 'fro');
      if (opt_disp ~= 0)
        disp(sprintf(['%4d | %10.5e \t| %10.5e \t| (%2d,%2d) \t|' ...
                      ' %10.5e \t| %10.5e \t| %10.5e \t|%10.5e \t|'], ...
                     times, norm_A2-f_k, norm_Lambda_k, i, j, c_k, h0, ...
                     sqrt(2)* abs(Lambda_k(i,j)), (f_k-f_k1)/abs(h0*c_k)));
      end 
      f_k1 = f_k;
      f_k = matr_sumdiag2(W_k);
      iter_pairs(times+1,:) = [i,j];
      iter_progress(times+1,:) = ...
          [norm_A2- f_k,norm_Lambda_k,c_k,h0,sqrt(2)* abs(Lambda_k(i,j))];   
      iter_times(times+1) = toc;
    end 
  end  
  
  info.iter = k;
  info.iter_pairs = iter_pairs;
  info.iter_progress = iter_progress;
  info.iter_times = iter_times;
  Uhat = U_k;
end


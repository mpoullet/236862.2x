function [X, A] = batch_thresholding(D, Y, epsilon)
% BATCH_THRESHOLDING Solve the pursuit problem via the error-constraint 
% Thresholding pursuit
% 
% Solves the following problem:
%   min_{alpha_i} \sum_i || alpha_i ||_0
%                  s.t.  ||y_i - D alpha_i||_2^2 \leq epsilon^2 for all i, 
% where D is a dictionary of size n X n, y_i are the input signals of
% length n (being the columns of the matrix Y) and epsilon stands 
% for the allowed residual error.
%
% The solution is returned in the matrix A, containing the representations 
% of the patches as its columns, along with the denoised signals
% given by  X = DA.
 
 
 
% TODO: Get the length of each atom
% Write your code here... n = ???;



% TODO: Get the number of patches
% Write your code here... m = ???;


 
% TODO: Compute the inner products between the dictionary atoms and the
% input patches (hint: the result should be a matrix of size n X N)
% Write your code here... inner_products = ???;


 
% Compute epsilon^2, which is the square residual error allowed per patch
epsilon_sq = epsilon^2;
 
% Compute the square value of each entry in 'inner_products' matrix
residual_sq = inner_products.^2;
 
% TODO: Sort each column in 'residual_sq' matrix in ascending order
% Write your code here... [mat_sorted, mat_inds] = sort(?, ?, ?);


 
% TODO: Compute the cumulative sums for each column of 'residual_sq'
% and save the result in the matrix 'accumulate_residual'
% Write your code here... accumulate_residual = ???;


 
% Compute the indices of the dominant coefficients that we want to keep
inds_to_keep = (accumulate_residual > epsilon_sq);
 
% TODO: Allocate a matrix of size n X N to save the sparse vectors
% Write your code here... A = ???;


 
% In what follows we compute the location of each non-zero to be assigned 
% to the matrix of the sparse vectors A. To this end, we need to map 
% 'mat_inds' to a linear subscript format. The mapping will be done using 
% Matlab's 'sub2ind' function.
 
% In order to use 'sub2ind' we need to compute the column subscript
col_sub = repmat(1:N,[n 1]);
 
% Now use 'sub2ind' to map 'mat_inds' to a linear subscript format.
linear_inds = sub2ind(size(A), mat_inds, col_sub);
 
% Reshape 'linear_inds' to a vector format
linear_inds = linear_inds(:);
 
% Map the entries in 'inds_to_keep' to their corresponding locations 
% in the matrix 'A' using the precomputed 'linear_inds' vector
linear_inds_to_keep = linear_inds(inds_to_keep(:));
 
% TODO: Assign to 'A' the coefficients in 'inner_products' using
% the precomputed vector 'linear_inds_to_keep'
% Write your code here... A( ??? ) = inner_products( ??? );


 
% Use 'sparse' format for the resulting sparse vectors
A = sparse(A);
 
% TODO: Reconstruct the patches using 'A' matrix
% Write your code here... X = ???;


 

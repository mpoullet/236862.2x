function [residual_error, avg_cardinality] = ...
    compute_stat(est_patches, orig_patches, est_coeffs)
% COMPUTE_STAT Compute and print usefull statistics of the pursuit and
% learning procedures
%
% Inputs:
%  est_patches  - A matrix, containing the recovered patches as its columns
%  orig_patches - A matrix, containing the original patches as its columns
%  est_coeffs   - A matrix, containing the estimated representations, 
%                 leading to est_patches, as its columns
%
% Outputs:
%  residual_error  - Average Mean Squared Error (MSE) per pixel
%  avg_cardinality - Average number of nonzeros that is used to represent 
%                    each patch
%

% Compute the Mean Square Error per patch
MSE_per_patch = sum((est_patches - orig_patches).^2 , 1);

% Compute the average 
residual_error = mean(MSE_per_patch)/size(orig_patches,1);

% Compute the average number of non-zeros
avg_cardinality = full(sum(abs(est_coeffs(:)) > 10^-10) / size(est_coeffs,2));

% Display the results
fprintf('Residual error %02.2f, Average cardinality %02.2f\n',...
    residual_error, avg_cardinality);

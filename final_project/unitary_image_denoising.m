function [est_unitary, D_unitary, mean_error, mean_cardinality] = ...
    unitary_image_denoising(noisy_im, D_init, num_learning_iterations, ...
    epsilon)
% UNITARY_IMAGE_DENOISING Denoise an image using unitary dictionary learning
%
% Inputs:
%   noisy_im - The input noisy image
%   D_init   - An initial UNITARY dictionary (e.g. DCT)
%   epsilon  - The noise-level in a PATCH,
%              used as the stopping criterion of the pursuit
%
% Outputs:
%   est_unitary - The denoised image
%   D_unitary   - The learned dictionary
%   mean_error  - A vector, containing the average representation error,
%                 computed per iteration and averaged over the total
%                 training examples
%   mean_cardinality - A vector, containing the average number of nonzeros,
%                      computed per iteration and averaged over the total
%                      training examples
%


%% Dictionary Learning

% TODO: Get the patch size [height, width] from D_init
% Write your code here... patch_size = ???;


% Divide the noisy image into fully overlapping patches
patches = im2col(noisy_im, patch_size, 'sliding');

% TODO: Train a dictionary via Procrustes analysis
% Write your code here... [D_unitary, mean_error, mean_cardinality] = unitary_dictionary_learning(?, ?, ?, ?);




%% Denoise the input image

% TODO: Step 1: Compute the representation of each noisy patch using the
% Thresholding pursuit
% Write your code here... [est_patches, est_coeffs] = batch_thresholding(?, ?, ?);



% TODO: Step 2: Reconstruct the image using 'col_to_im' function
% Write your code here... est_unitary = col_to_im(?, ?, ?);




%% Compute and display the statistics

fprintf('\n\nUnitary dictionary: ');
compute_stat(est_patches, patches, est_coeffs);


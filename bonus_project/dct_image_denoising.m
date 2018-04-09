function est_dct = dct_image_denoising(noisy_im, D_DCT, epsilon)
% DCT_IMAGE_DENOISING Denoise an image via the DCT transform
%
% Inputs:
%   noisy_im - The input noisy image
%   D_DCT    - A column-normalized DCT dictionary
%   epsilon  - The noise-level of the noise in a PATCH,
%              used as the stopping criterion of the pursuit
%
% Output:
%   est_dct - The denoised image
%

% Get the patch size [height, width] from D_DCT
patch_size = sqrt(size(D_DCT));

% Divide the noisy image into fully overlapping patches
patches = im2col(noisy_im, patch_size, 'sliding');

% Step 1: Compute the representation of each noisy patch using the
% Thresholding pursuit
[est_patches, est_coeffs] = batch_thresholding(D_DCT, patches, epsilon);

% Step 2: Reconstruct the image using 'col_to_im' function
est_dct = col_to_im(est_patches, patch_size, size(noisy_im));

% Compute and display the statistics
fprintf('DCT dictionary: ');
compute_stat(est_patches, patches, est_coeffs);

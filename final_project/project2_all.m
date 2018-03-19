clear; clc; close all;

%% Part A: Data construction

% Read an image
orig_im = imread('barbara.png');
% Crop the image
orig_im = orig_im(1:1+255, 251:251+255);
% Convert to double
orig_im = double(orig_im);

% Set the seed for the random generator
seed = 7;

% Set a fixed random seed to reproduce the results
rng(seed);
 
% Set the standard-deviation of the Gaussian noise
sigma = 20;

% Add noise to the input image
noisy_im = orig_im + sigma*randn(size(orig_im));

% Compute the PSNR of the noisy image and print its value
psnr_noisy = compute_psnr(orig_im, noisy_im);
fprintf('PSNR of the noisy image is %.3f\n\n\n', psnr_noisy);

% Show the original and noisy images
figure(1);
subplot(2,2,1); imshow(orig_im,[]);
title('Original');
subplot(2,2,2); imshow(noisy_im,[]);
title(['Noisy, PSNR = ' num2str(psnr_noisy)]);

% For the report
figure(4);
imshow(orig_im,[]);
title('Original');
print('Cropped_Barbara_image', '-depsc');

figure(5);
imshow(noisy_im,[]);
title(['Noisy, PSNR = ' num2str(psnr_noisy)]);
print('Noisy_Barbara_image', '-depsc');

%% Part B: Patch-Based Image Denoising
 
% Set the patch dimensions [height, width]
patch_size = [10 10];
 
% Create a unitary DCT dictionary
D_DCT = build_dct_unitary_dictionary(patch_size);

% Show the unitary DCT dictionary
figure(2);
subplot(1,2,1); show_dictionary(D_DCT);
title('Unitary DCT Dictionary');

%% Part B-1: Unitary DCT denoising
 
% Set the noise-level of a PATCH for the pursuit,
% multiplied by some constant (e.g. sqrt(1.1)) to improve the restoration
epsilon_dct = sqrt(1.1)*10*sigma;
 
% Denoise the input image via the DCT transform
est_dct = dct_image_denoising(noisy_im, D_DCT, epsilon_dct);
 
% Compute and print the PSNR
psnr_dct = compute_psnr(orig_im, est_dct);
fprintf('DCT dictionary: PSNR %.3f\n\n\n', psnr_dct);
 
% Show the resulting image
figure(1);
subplot(2,2,3); imshow(est_dct,[]);
title(['DCT: \epsilon = ' num2str(epsilon_dct) ' PSNR = ' num2str(psnr_dct)]);

% For the report
figure(6);
imshow(est_dct,[]);
title(['DCT: \epsilon = ' num2str(epsilon_dct) ' PSNR = ' num2str(psnr_dct)]);
print('DCT_reconstructed_image', '-depsc');
 
%% Part B-2: Unitary dictionary learning for image denoising
 
% Set the number of training iterations for the learning algorithm
T = 20;

% Set the noise-level of a PATCH for the pursuit,
% multiplied by some constant (e.g. sqrt(1.1)) to improve the restoration
epsilon_learning = epsilon_dct;

% Denoise the image using unitary dictionary learning
[est_learning, D_learned, mean_error, mean_cardinality] = ...
    unitary_image_denoising(...
    noisy_im, D_DCT, T, epsilon_learning);

% Show the dictionary
figure(2);
subplot(1,2,2); show_dictionary(D_learned);
title('Learned Unitary Dictionary');

% For the report
figure(7);
show_dictionary(D_learned);
title('Learned Unitary Dictionary');
print('Learned Unitary Dictionary', '-depsc');

% Show the representation error and the cardinality as a function of the
% learning iterations
figure(3);
subplot(1,2,1); plot(1:T, mean_error, 'linewidth', 2);
ylim([0 5*sqrt(prod(patch_size))*sigma]);
ylabel('Average Representation Error');
xlabel('Learning Iteration');
subplot(1,2,2); plot(1:T, mean_cardinality, 'linewidth', 2);
ylabel('Average Number of Non-Zeros');
xlabel('Learning Iteration');

% For the report
print('Average_Representation_Error_and_Cardinality', '-depsc');

% Compute and print the PSNR
psnr_unitary = compute_psnr(orig_im, est_learning);
fprintf('Unitary dictionary: PSNR %.3f\n\n\n', psnr_unitary);

% Show the results
figure(1);
subplot(2,2,4); imshow(est_learning,[]);
title(['Unitary: \epsilon = ' num2str(epsilon_learning) ' PSNR = ' num2str(psnr_unitary)]);

% For the report
figure(8);
imshow(est_learning,[]);
title(['Unitary: \epsilon = ' num2str(epsilon_learning) ' PSNR = ' num2str(psnr_unitary)]);
print('Procrustes_reconstructed_image', '-depsc');

%% SOS Boosting

% Set the strengthening factor
% Typical choice, 0<rho<=1
rho = 1;

% Set the noise-level in a PATCH for the pursuit. 
% A common choice is a slightly higher noise-level than the one set 
% in epsilon_learning, e.g. 1.1*epsilon_learning;
epsilon_sos = 1.1*epsilon_learning;

% Init D_sos to be D_learned 
D_sos = D_learned;

% Compute the signal-strengthen image by adding to 'noisy_im' the 
% denoised image 'est_learning', multiplied by an appropriate 'rho'
s_im = noisy_im + rho*est_learning;

% Operate the denoiser on the signal-strengthened image
est_learning_sos = unitary_image_denoising(s_im, D_sos, T, epsilon_sos);

% Subtract from 'est_learning_sos' image the previous estimate
% 'est_learning', multiplied by the strengthening factor
est_learning_sos = est_learning_sos - rho*est_learning;

% Compute and print the PSNR
psnr_unitary_sos = compute_psnr(orig_im, est_learning_sos);
fprintf('SOS Boosting: epsilon %.3f, rho %.2f, PSNR %.3f\n\n\n', ...
    epsilon_sos, rho, psnr_unitary_sos);
 
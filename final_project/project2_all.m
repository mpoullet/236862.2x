clear; clc; close all;
 
 
%% Part A: Data construction
 
% Read an image
orig_im = imread('barbara.png');
% Crop the image
orig_im = orig_im(1:1+256, 251:251+256);
% Convert to double
orig_im = double(orig_im);

% TODO: Set the seed for the random generator
% Write your code here... seed = ???;


 
% Set a fixed random seed to reproduce the results
rng(seed);
 
% TODO: Set the standard-deviation of the Gaussian noise
% Write your code here... sigma = ???;


 
% TODO: Add noise to the input image
% Write your code here... noisy_im = ???;




% Compute the PSNR of the noisy image and print its value
psnr_noisy = compute_psnr(orig_im, noisy_im);
fprintf('PSNR of the noisy image is %.3f\n\n\n', psnr_noisy);
 
% Show the original and noisy images
figure (1);
subplot(2,2,1); imshow(orig_im,[]);
title('Original');
subplot(2,2,2); imshow(noisy_im,[]);
title(['Noisy, PSNR = ' num2str(psnr_noisy)]);
 
 
%% Part B: Patch-Based Image Denoising
 
% Set the patch dimensions [height, width]
patch_size = [10 10];
 
% TODO: Create a unitary DCT dictionary
% Write your code here... D_DCT = build_dct_unitary_dictionary( ? );


 
% Show the unitary DCT dictionary
figure(2);
subplot(1,2,1); show_dictionary(D_DCT);
title('Unitary DCT Dictioanry');
 
 
%% Part B-1: Unitary DCT denoising
 
% TODO: Set the noise-level of a PATCH for the pursuit,
% multiplied by some constant (e.g. sqrt(1.1)) to improve the restoration
% Write your code here... epsilon_dct = ???;


 
% Denoise the input image via the DCT transform
est_dct = dct_image_denoising(noisy_im, D_DCT, epsilon_dct);
 
% Compute and print the PSNR
psnr_dct = compute_psnr(orig_im, est_dct);
fprintf('DCT dictionary: PSNR %.3f\n\n\n', psnr_dct);
 
% Show the resulting image
figure(1);
subplot(2,2,3); imshow(est_dct,[]);
title(['DCT: \epsilon = ' num2str(epsilon_dct) ' PSNR = ' num2str(psnr_dct)]);

 
%% Part B-2: Unitary dictionary learning for image denoising
 
% TODO: Set the number of training iterations for the learning algorithm
% Write your code here... T = ???;


 
% TODO: Set the noise-level of a PATCH for the pursuit,
% multiplied by some constant (e.g. sqrt(1.1)) to improve the restoration
% Write your code here... epsilon_learning = ???;


 
% Denoise the image using unitary dictionary learning
[est_learning, D_learned, mean_error, mean_cardinality] = ...
    unitary_image_denoising(...
    noisy_im, D_DCT, T, epsilon_learning);
 
% Show the dictionary
figure(2);
subplot(1,2,2); show_dictionary(D_learned);
title('Learned Unitary Dictioanry');
 
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
 
% Compute and print the PSNR
psnr_unitary = compute_psnr(orig_im, est_learning);
fprintf('Unitary dictionary: PSNR %.3f\n\n\n', psnr_unitary);
 
% Show the results
figure(1);
subplot(2,2,4); imshow(est_learning,[]);
title(['Unitary: \epsilon = ' num2str(epsilon_learning) ' PSNR = ' num2str(psnr_unitary)]);

 
%% SOS Boosting
 
% TODO: Set the strengthening factor
% Write your code here... rho = ???;


 
% TODO: Set the noise-level in a PATCH for the pursuit. 
% A common choice is a slightly higher noise-level than the one set 
% in epsilon_learning, e.g. 1.1*epsilon_learning;
% Write your code here... epsilon_sos = ???;



% TODO: Init D_sos to be D_learned 
% Write your code here... D_sos = ???;


 
% TODO: Compute the signal-strengthen image by adding to 'noisy_im' the 
% denoised image 'est_learning', multiplied by an appropriate 'rho'
% Write your code here... s_im = ???;


 
% TODO: Operate the denoiser on the signal-strengthened image
% Write your code here... est_learning_sos = unitary_image_denoising(?, ?, ?, ?);



% TODO: Subtract from 'est_learning_sos' image the previous estimate
% 'est_learning', multiplied by the strengthening factor
% Write your code here... est_learning_sos = ???;


 
% Compute and print the PSNR
psnr_unitary_sos = compute_psnr(orig_im, est_learning_sos);
fprintf('SOS Boosting: epsilon %.3f, rho %.2f, PSNR %.3f\n\n\n', ...
    epsilon_sos, rho, psnr_unitary_sos);
 

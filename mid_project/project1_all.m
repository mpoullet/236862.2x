clear; clc; close all;

%% Part A: Data Construction and Parameter-Setting

% Read an image
im = imread('barbara.png'); 
% Convert to double
im = double(im); 

% Show the image
figure; imshow(im,[]); title('Original image');
print('Original_image', '-depsc');

% Patch dimensions [height, width]
patch_size = [6 6];

% Divide the image into FULLY overlapping patches using Matlab's
% function 'im2col'.
all_patches = im2col(im, patch_size);

% Number of patches to train on
num_train_patches = 10000; 
% Number of patches to test on
num_test_patches = 5000; 

% Set the seed for the random generator
seed = 7;
 
% Set a fixed random seed to reproduce the results
rng(seed);

% Create a training set by choosing a random subset 
% of 'num_train_patches' taken from 'all_patches'
[n,m]=size(all_patches);
permutation = randperm(m);
train_patches = all_patches(:,permutation(1:num_train_patches));

% Create a test set by choosing another random subset 
% of 'num_test_patches' taken from the remaining patches
test_patches = all_patches(:,permutation(num_train_patches+1:num_train_patches+num_test_patches));

% Initialize the dictionary
D_DCT = build_dct_unitary_dictionary(patch_size);

% Show the unitary DCT dictionary

% For the report
figure(2);
show_dictionary(D_DCT);
title('Unitary DCT Dictionary');
print('Unitary_DCT_Dictionary', '-depsc');

figure(3);
subplot(1,2,1); show_dictionary(D_DCT);
title('Unitary DCT Dictionary');
 
% Set K - the cardinality of the solution.
% This will serve us later as the stopping criterion of the pursuit
K = 4;

%% Part B: Compute the Representation Error Obtained by the DCT Dictionary
 
% Compute the representation of each patch that belongs to the training set
% using Thresholding
[est_train_patches_dct, est_train_coeffs_dct] = ...
    batch_thresholding(D_DCT, train_patches, K);
 
% Compute the representation of each patch that belongs to the test set
% using Thresholding
[est_test_patches_dct, est_test_coeffs_dct] = ...
    batch_thresholding(D_DCT, test_patches, K);
 
% Compute and display the statistics
fprintf('\n\nDCT dictionary: Training set, ');
compute_stat(est_train_patches_dct, train_patches, est_train_coeffs_dct);
fprintf('DCT dictionary: Testing  set, ');
compute_stat(est_test_patches_dct, test_patches, est_test_coeffs_dct);
fprintf('\n\n');
 
%% Part C: Procrustes Dictionary Learning
 
% Set the number of training iterations
T = 20;

% Train a dictionary via Procrustes analysis
[D_learned, mean_error, mean_cardinality] = ...
    unitary_dictionary_learning(train_patches, D_DCT, T, K);
 
% Show the dictionary
figure(3);
subplot(1,2,2); show_dictionary(D_learned);
title('Learned Unitary Dictionary');

% For the report
figure(4);
show_dictionary(D_learned);
title('Learned Unitary Dictionary');
print('Learned_Unitary_Dictionary','-depsc');
 
% Show the representation error and the cardinality as a function of the
% learning iterations
figure(5);
subplot(1,2,1); plot(1:T, mean_error, 'linewidth', 2);
ylabel('Average Representation Error');
xlabel('Learning Iteration');
subplot(1,2,2); plot(1:T, mean_cardinality, 'linewidth', 2);
ylabel('Average Number of Non-Zeros'); ylim([K-1 K+1]);
xlabel('Learning Iteration');
print('Average_MSE_Non-Zeros_vs_Iterations', '-depsc');
 
% Compute the representation of each signal that belong to the training set
% using Thresholding
[est_train_patches_learning, est_train_coeffs_learning] = ...
    batch_thresholding(D_learned, train_patches, K);
 
% Compute the representation of each signal that belong to the testing set
% using Thresholding
[est_test_patches_learning, est_test_coeffs_learning] = ...
    batch_thresholding(D_learned, test_patches, K);
 
% Compute and display the statistics
fprintf('\n\nLearned dictionary: Training set, ');
compute_stat(est_train_patches_learning, train_patches, ...
    est_train_coeffs_learning);
fprintf('Learned dictionary: Testing  set, ');
compute_stat(est_test_patches_learning, test_patches, ...
    est_test_coeffs_learning);
fprintf('\n\n');

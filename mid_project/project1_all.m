clear; clc; close all;


%% Part A: Data Construction and Parameter-Setting

% Read an image
im = imread('barbara.png'); 
% Convert to double
im = double(im); 

% Show the image
figure; imshow(im,[]); title('Original image');

% Patch dimensions [height, width]
patch_size = [6 6];

% TODO: Divide the image into FULLY overlapping patches using Matlab's
% function 'im2col'.
% Write your code here... all_patches = ???;


% Number of patches to train on
num_train_patches = 10000; 
% Number of patches to test on
num_test_patches = 5000; 

% TODO: Set the seed for the random generator
% Write your code here... seed = ???;

 
% Set a fixed random seed to reproduce the results
rng(seed);

% TODO: Create a training set by choosing a random subset 
% of 'num_train_patches' taken from 'all_patches'
% Write your code here... train_patches = ???;


 
% TODO: Create a test set by choosing another random subset 
% of 'num_test_patches' taken from the remaining pathces
% Write your code here... test_patches = ???;


 
% TODO: Initialize the dictionary
% Write your code here... D_DCT = build_dct_unitary_dictionary( ? );



% Show the unitary DCT dictionary
figure(2);
subplot(1,2,1); show_dictionary(D_DCT);
title('Unitary DCT Dictionary');
 
% TODO: Set K - the cardinality of the solution.
% This will serve us later as the stopping criterion of the pursuit
% Write your code here... K = ???;


 
 
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
 
% TODO: Set the number of training iterations
% Write your code here... T = ???;


 
% Train a dictionary via Procrustes analysis
[D_learned, mean_error, mean_cardinality] = ...
    unitary_dictionary_learning(train_patches, D_DCT, T, K);
 
% Show the dictionary
figure(2);
subplot(1,2,2); show_dictionary(D_learned);
title('Learned Unitary Dictionary');
 
% Show the representation error and the cardinality as a function of the
% learning iterations
figure(3);
subplot(1,2,1); plot(1:T, mean_error, 'linewidth', 2);
ylabel('Average Representation Error');
xlabel('Learning Iteration');
subplot(1,2,2); plot(1:T, mean_cardinality, 'linewidth', 2);
ylabel('Average Number of Non-Zeros'); ylim([K-1 K+1]);
xlabel('Learning Iteration');
 
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
 

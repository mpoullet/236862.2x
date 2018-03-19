clear; clc; close all;

%% Part A: Data Construction

% Read an image
orig_im = imread('foreman.png');
% Convert to double
orig_im = double(orig_im);

% Define the blur kernel
h_blur = ones(9)/81;
% Blur the image
blurred_im = imfilter(orig_im,h_blur,'circular');

% Set the global noise standard-deviation
sigma = sqrt(2);
% Add noise to the blurry image
y = blurred_im + sigma*randn(size(orig_im));

% Compute and print the PSNR of the degraded image
psnr_input = compute_psnr(orig_im, y);
fprintf('PSNR of the blurred and noisy image is %.3f\n\n\n', psnr_input);

% Show the original and degraded images
figure(1);
subplot(1,3,1); imshow(orig_im,[]);
title('Original');
subplot(1,3,2); imshow(y,[]);
title(['Input: PSNR = ' num2str(psnr_input)]);

% For the report
figure(3);
imshow(orig_im,[]);
title('Original');
print('Original_image', '-depsc');

figure(4);
imshow(blurred_im,[]);
title('Blurred');
print('Blurred_image', '-depsc');

figure(5);
imshow(y,[]);
title(['Blurred and noisy: PSNR = ' num2str(psnr_input)]);
print('Blurred_and_noisy', '-depsc');

%% Part B: Deblurring via Regualrization by Denoising (RED)

% In this part we aim to tackle the image deblurring problem.
% To this end, we will use RED, suggest minimizing
% \min_x 1/(2*sigma^2) * ||H*x - y ||_2^2 + lambda/2 * x'*(x - f(x)),
% where f(x) is a denoising algorithm that operates on x, and the matrix
% H is a known blur kernel.
% We will use the Fixed-Point scheme to minimize the above objective, where
% the update rule is given by
% x_new = [1/sigma^2 H'*H + lambda*I]^(-1) * [1/sigma^2 * H'*y + lambda * f(x)]

% Choose the parameters of the DCT denoiser

% Set the patch dimensions [height, width]
patch_size = [6 6];

% Create a unitary DCT dictionary
D = build_dct_unitary_dictionary(patch_size);

% Set the noise-level in a PATCH for the pursuit
epsilon = sqrt(1.1)*6*sigma;

% Set the number of outer FP iterations
K = 50;

% Set the number of inner iterations for solving the linear system Az=b
% using Richardson algorithm
T = 50;

% Set the step size of each step of the Richardson method
mu = 1;

% Set the FP parameter lambda that controls the importance of the prior
lambda = 0.1;

% Allocate a vector that stores the PSNR per iteration
psnr_red = zeros(K,1);

% Initialize the solution x with the input image
x = y;

% Run the fixed-point algorithm for num_outer_iters
for outer_iter = 1:K

    % Stage 1:
    
    % Compute fx = f(x_prev), by running our DCT image denoising 
    % algorithm
    fx = dct_image_denoising(x, D, epsilon);

    % Stage 2:
    
    % Solve the linear system Az=b, where
    % A = 1/sigma^2*H'H + lambda*I, and b = 1/sigma^2*H'y + lambda*x.
    % This is done using the (iterative) Richardson method,
    % where its update rule is given by
    % z_new = z_old - mu*(A*z_old - b)
    
    % Initialize the the previous solution 'z_old' of the Richardson
    % method to be the denoised image 'fx'
    z_old = fx;

    % Compute b = 1/sigma^2*H'y + lambda*x. The multiplication
    % by H or H' is done by filtering the image we operate on with
    % 'h_blur'. Notice that H is symmetric, therefore the multiplication by
    % H or H' is done in the very same way.
    HT_y = imfilter(y, h_blur, 'circular');
    b = 1/sigma^2 * HT_y + lambda*fx;

    % Repeat z_new = z_old - step_size*(A*z_old - b) for num_inner_iters
    step_size = 1;
    for inner_iter = 1:T

        % Compute H*z_old by convolving the image 'z_old' with the 
        % filter 'h_blur'
        H_z_old = imfilter(z_old,h_blur,'circular');
        
        % Compute H'*H_z_old by convolving the image 'H_z_old' with 
        % the filter 'h_blur' (in our case H is symmetric)
        HTH_z_old = imfilter(H_z_old,h_blur,'circular');
        
        % Compute the image A*z_old which is nothing but
        % 1/sigma^2*H'*H*z_old + lambda*z_old
        A_z_old = 1/sigma^2*HTH_z_old + lambda*z_old;
        
        % Compute z_new = z_old - step_size*(A*z_old - b)
        z_new = z_old - step_size*(A_z_old - b);
        
        % Update z_old to be the new z
        z_old = z_new;
        
    end

    % Update x to be z_new
    x = z_new;
    
    % Compute the PSNR of the restored image
    psnr_red(outer_iter) = compute_psnr(orig_im, x);
    fprintf('RED: Fixed-Point Iter %02d, PSNR %.3f\n\n', ...
        outer_iter, psnr_red(outer_iter));
end

%% Present the results

% Show the restored image obtained by RED
figure(1);
subplot(1,3,3); imshow(x, []);
title(['RED: PSNR = ' num2str(psnr_red(end))]);

% For the report
figure(6);
imshow(x, []);
title(['RED: PSNR = ' num2str(psnr_red(end))]);
print('RED_reconstructed_image', '-depsc');

% Plot the PSNR of the RED as a function of the iterations
figure(2); plot(1:K, psnr_red); grid on;
title('RED: PSNR vs. Iterations');
ylabel('PSNR');
xlabel('Fixed-Point Iteration');
print('PSNR_vs_Iterations', '-depsc');

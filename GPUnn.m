load('mnist-testing.mat');
load('mnist-training.mat');

%% Define Functions for Training
% activation function ReLU and its derivative
ReLU = @(x) max(0, x);
dReLU = @(x) x > 0;
% final activation function
softmax = @(x) exp(x - max(x, [], 1)) ./ sum(exp(x - max(x, [], 1)), 1);

%% Prepare the Data
% format y to be a one-hot encoding of the labels
y = gpuArray(zeros(10, size(trainLabels, 1)));
for i = 1:size(trainLabels, 1)
    y(trainLabels(i, 1) + 1, i) = 1;
end

% reshape input vectors for easier feature extraction
x = gpuArray(zeros(784, 24000));
for i = 1:24000
    x(:, i) = reshape(trainImages(:, :, i), 784, 1);
end

% randomly initialize weights and biases between -0.5 and 5
w_1 = gpuArray(rand(16, 784) - 0.5);
w_2 = gpuArray(rand(10, 16) - 0.5);
b_1 = gpuArray(rand(16, 1) - 0.5);
b_2 = gpuArray(rand(10, 1) - 0.5);

epochs = 500; % number of iterations of forward and back propogation
n = size(trainLabels, 1);
alpha = 0.1; % learning rate
loss = zeros(epochs, 1);

%% Train the Model
for i = 1:epochs
    % forward pass
    z_1 = w_1 * x + b_1 * ones(1, size(x, 2));
    a_1 = ReLU(z_1);
    z_2 = w_2 * a_1 + b_2 * ones(1, size(a_1, 2));
    a_2 = softmax(z_2);

    loss(i, 1) = norm(a_2 - y)^2; % square-error calculation

    % gradient calculation
    dz_2 = 2 * (a_2 - y);
    dw_2 = dz_2 * a_1' / n;
    db_2 = sum(dz_2, 2) / n;

    dz_1 = (w_2' * dz_2) .* dReLU(z_1);
    dw_1 = dz_1 * x' / n;
    db_1 = sum(dz_1, 2) / n;

    % back propagation
    w_2 = w_2 - alpha * dw_2;
    w_1 = w_1 - alpha * dw_1;
    b_2 = b_2 - alpha * db_2;
    b_1 = b_1 - alpha * db_1;
end

% plot loss over Time
figure(1);
plot(loss, 'Color', 'red');
title('Cost of Function Over Epochs');
xlabel('Epoch');
ylabel('Cost');
grid on;

% transfer model from GPU to CPU and save
w_1 = gather(w_1);
w_2 = gather(w_2);
b_1 = gather(b_1);
b_2 = gather(b_2);

save network.mat w_1 w_2 b_1 b_2;
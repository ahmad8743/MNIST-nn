function b = classifier(img)
    load('network.mat')
    ReLU = @(x) max(0, x);
    softmax = @(x) exp(x - max(x, [], 1))./sum(exp(x - max(x, [], 1)), 1);

    l = reshape(img, 784, 1);
    z_1 = w_1 * l + b_1;
    a_1 = ReLU(z_1);
    z_2 = w_2 * a_1 + b_2;
    o = softmax(z_2);

    [~, b] = max(o);
    b = b - 1;
end
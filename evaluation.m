function [accuracy, confusion_matrix] = evaluation()
    load('mnist-testing.mat')
    load('network.mat')
    count = 0;
    con = zeros(10, 10);
    
    for i = 1:8000
        sample = classifier(testImages(:, :, i));
        actual = testLabels(i, 1);
        if sample == actual
            count = count + 1;
        end
        con(actual + 1, sample + 1) = con(actual + 1, sample + 1) + 1;
       
    end
    
    accuracy = count / 8000 * 100;
    confusion_matrix = con;
end
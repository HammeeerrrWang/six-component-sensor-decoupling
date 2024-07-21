%双阶段解耦 bp分类+二次项拟合 

warning off 
close all   
clear       
clc         


predict_data = readtable('回归验证集.xlsx');
X_predict = predict_data{:, 14:29};   
X_predict_scaled = rescale(X_predict, 0, 1); 
y_true = predict_data{:, 8:13};


model_classify = load('bp_classify_model_patternnet.mat');
net_classify = model_classify.net;


predictions_classify = net_classify(X_predict_scaled');
predicted_labels = vec2ind(predictions_classify)'; 


regression_predictions = zeros(size(X_predict, 1), 6); 

% 对每个预测的标签进行循环
for i = 1:length(predicted_labels)
    label = predicted_labels(i);
    binary_label = dec2bin(label, 6) - '0'; 

    for j = 1:3
        if binary_label(j) == 1
        
            modelFileName = fullfile('bp_regression_model', ['bp_regression_model_' num2str(label)]);
            model_regression = load(modelFileName);
            net_regression = model_regression.net;

 
            bp_output = net_regression(X_predict(i, :)');
            regression_predictions(i, j) = bp_output(j);
        else
            regression_predictions(i, j) = 0;
        end
    end

    % 遍历二进制标签的后三位
    for j = 4:6
        if binary_label(j) == 1

            modelFileName = fullfile('quadratictermfitting_regression_model', ['qt_regression_' num2str(label) '_' num2str(j) '.mat']);
            model_regression = load(modelFileName);
            lm = model_regression.lm;

            regression_predictions(i, j) = predict(lm, X_predict(i, :));
        else
            regression_predictions(i, j) = 0; 
        end
    end
end



% 显示回归预测结果
disp(regression_predictions);


for i = 1:6 
    
    non_zero_indices = y_true(:, i) ~= 0;
    if sum(non_zero_indices) > 0
        mape_values(i) = mean(abs((y_true(non_zero_indices, i) - regression_predictions(non_zero_indices, i)) ./ y_true(non_zero_indices, i))) * 100;
    else
        mape_values(i) = NaN; 
    end

    
    figure; 
    plot(1:length(y_true(:, i)), y_true(:, i), 'r-*', ...
         1:length(regression_predictions(:, i)), regression_predictions(:, i), 'b-o', ...
         'LineWidth', 1);
    legend('True value', 'Regression value', 'FontSize', 22, 'FontName', 'Times New Roman');
    xlabel('Sample serial number', 'FontSize', 22, 'FontName', 'Times New Roman');
    ylabel(['Output ', num2str(i)], 'FontSize', 22, 'FontName', 'Times New Roman');
    title(['Two-stage decoupling output channel ', num2str(i), ' regression results'], 'FontSize', 22, 'FontName', 'Times New Roman');
    set(gca, 'FontSize', 22, 'FontName', 'Times New Roman');
    ylim([-750 4500]); 
    yticks(-750:750:4500); 
    ytickformat('%,.0f'); 
    grid on; 
end




disp('MAPE for each output:');
disp(mape_values);



%bp神经网网络分类 
warning off 
close all   
clear      
clc         


train_data = readtable('训练集.xlsx');
test_data = readtable('分类验证集.xlsx');


X_train = train_data{:, 14:29};  
y_train = train_data{:,1}; 

X_test = test_data{:, 14:29};   
y_test = test_data{:,1};   
B = size(X_test,1);

 
X_train_scaled = rescale(X_train, 0, 1);
X_test_scaled = rescale(X_test, 0, 1);

 
numClasses = 64;  
T_train = full(ind2vec(y_train', numClasses));
T_test = full(ind2vec(y_test', numClasses));

 
hiddenLayerSize = [100,80,50];
net = patternnet(hiddenLayerSize, 'trainscg');  

 
net.trainParam.epochs = 4000;
net.trainParam.lr = 0.001;
net.trainParam.max_fail = 2000; 


[net, tr] = train(net, X_train_scaled', T_train);


save('bp_classify_model_patternnet', 'net')


predictions = net(X_test_scaled');
predictions = vec2ind(predictions)'; 

predicted_labels_double = double(predictions);


accuracy = sum(y_test == predictions) / numel(y_test);
fprintf('Accuracy: %f\n', accuracy);



B = numel(y_test);  
accuracy = sum(y_test == predictions) / numel(y_test);  
accuracyStr = num2str(accuracy * 100, '%.2f');  



figure;
plot(1:B, y_test, 'r-*', 1:B, predicted_labels_double, 'b-o', 'LineWidth', 1);
legend('True label', 'Predicted label', 'FontSize', 22 ,'FontName', 'Times New Roman');
xlabel('Sample serical number','FontName', 'Times New Roman','FontSize', 22);
ylabel('Label value','FontName', 'Times New Roman','FontSize', 2);
title(['Classification results (accuracy: ', num2str(accuracy * 100, '%.2f'), '%)'],'FontName', 'Times New Roman','FontSize', 22);
set(gca, 'FontSize', 22, 'FontName', 'Times New Roman');
xlim([1, B]);
grid on;


C = confusionmat(y_test, predictions);


figure;
confusionchart(C);
title('混淆矩阵');


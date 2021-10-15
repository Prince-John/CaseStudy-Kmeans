%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace


% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.
% TODO: NEED TO HAVE centroid_labels

predictions = zeros(200,1);
outliers = zeros(200,1);

% loop through the test set, figure out the predicted number
for i = 1:200

testing_vector=test(i,:);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);

predictions(i) = centroid_labels(prediction_index);

end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
sum_all_rows = sum(train(:,1:784));
outliers_filter = isoutlier(sum_all_rows);
inliers_filter = ~outliers_filter;

not_match_count_rows = zeros(size(test,1),1);

for i=1:200
    not_match_count_row = 0;
    for j=1:784
        if(outliers_filter(j) == 0 && test(i,j) ~= 0)
            not_match_count_row = not_match_count_row + 1;
        end
    end
%     Convert this array to row major order to increase efficency of cache
    not_match_count_rows(i) = not_match_count_row;
end

% Uncommemt this to use the method that classify outlier to have the
% mismatch count to be bigger than 95% of all elements in
% "not_match_count_rows" array
% quantile_stats = quantile(not_match_count_rows,0.95);
% count_of_outliers = sum(not_match_count_rows > quantile_stats);
% % colNrs is an array contains indices of predicted outliers
% colNrs = find(not_match_count_rows >= quantile_stats);


% We used isoutlier function given by matlab to detect outliers within the
% "not_match_count_rows" array
outliers_ = isoutlier(not_match_count_rows);
% colNrs is an array contains indices of predicted outliers
colNrs = find(outliers_ > 0);
count_of_outliers = length(colNrs);

figure;
colormap('gray');

plotsize_outliers = ceil(sqrt(length(colNrs)));

for ind=1:length(colNrs)
    
    outlier=test(colNrs(ind),(1:784));
    subplot(plotsize,plotsize,ind);
    
    imagesc(reshape(outlier,[28 28])');
    title(strcat('Outlier',num2str(ind)))

end

% Identify outliers to be image 3 (image 56th) and image 9 (image 187th). 
% Other images are doubtfully to be outliers. Surprisingly, the two methods
% confirmed that image 9 are outliers
    
%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
% FILL IN
stem(outliers_);
title("Outlier stem plot");

%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o');
hold on;
plot(predictions,'x');
title('Predictions');

%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(correctlabels==predictions)

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
% FILL IN
    norms = zeros(size(centroids,1),1);
    for i = 1:size(centroids,1)
        norms(i) = norm(data -centroids(i,:))^2;
    end
    [vec_distance, index] = min(norms);
    return;
end


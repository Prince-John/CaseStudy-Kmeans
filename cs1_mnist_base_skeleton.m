close all;
%%

%c0 = reshape(imread('generated_centroids/0.bmp').', 1, []);


%% In this script, you need to implement three functions as part of the k-means algorithm.
% These steps will be repeated until the algorithm converges:

  % 1. initialize_centroids
  % This function sets the initial values of the centroids
  
  % 2. assign_vector_to_centroid
  % This goes through the collection of all vectors and assigns them to
  % centroid based on norm/distance
  
  % 3. update_centroids
  % This function updates the location of the centroids based on the collection
  % of vectors (handwritten digits) that have been assigned to that centroid.


%% Initialize Data Set
% These next lines of code read in two sets of MNIST digits that will be used for training and testing respectively.

% training set (1500 images)
train=csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train=train(:,1:784);
train(:,785)=zeros(1500,1);

% testing set (200 images with 11 outliers)
test=csvread('mnist_test_200_woutliers.csv');
% store the correct test labels
correctlabels = test(:,785);
test=test(:,1:784);

% now, zero out the labels in "test" so that you can use this to assign
% your own predictions and evaluate against "correctlabels"
% in the 'cs1_mnist_evaluate_test_set.m' script
test(:,785)=zeros(200,1);

%% After initializing, you will have the following variables in your workspace:
% 1. train (a 1500 x 785 array, containins the 1500 training images)
% 2. test (a 200 x 785 array, containing the 200 testing images)
% 3. correctlabels (a 200 x 1 array containing the correct labels (numerical
% meaning) of the 200 test images

%% To visualize an image, you need to reshape it from a 784 dimensional array into a 28 x 28 array.
% to do this, you need to use the reshape command, along with the transpose
% operation.  For example, the following lines plot the first test image

figure;
colormap('gray'); % this tells MATLAB to depict the image in grayscale
testimage = reshape(test(1,(1:784)), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.

%% After importing, the array 'train' consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.

%% This next section of code calls the three functions you are asked to specify

k= 10; % set k
max_iter= 20; % set the number of iterations of the algorithm

%% The next line initializes the centroids.  Look at the initialize_centroids()
% function, which is specified further down this file.

centroids=initialize_centroids(train,k);

%% Initialize an array that will store k-means cost at each iteration

cost_iteration = zeros(max_iter, 1);

%% This for-loop enacts the k-means algorithm

training_size = size(train,1);

for iter=1:max_iter
    
    %assinging centroids to all the vectors
    for i = 1:training_size
        [centroid_index, dist] = assign_vector_to_centroid(train(i,:),centroids);
        train(i,785)= centroid_index;
        cost_iteration(iter) = cost_iteration(iter)+ dist;
    end
    cost_iteration(iter) = cost_iteration(iter)/training_size;  
    
    centroids = update_Centroids(train, k);
end

%% This section of code plots the k-means cost as a function of the number
% of iterations

figure;
plot(cost_iteration); 


%% This next section of code will make a plot of all of the centroids
% Again, use help <functionname> to learn about the different functions
% that are being used here.

figure;
colormap('gray');

plotsize = ceil(sqrt(k));


for ind=1:k
    
    centr=centroids(ind,(1:784));
    subplot(plotsize,plotsize,ind);
    
    imagesc(reshape(centr,[28 28])');
    title(strcat('Centroid ',num2str(ind)))

end

centroid_labels = [1,3,4,0,8,6,7,9,0,2];
%% Testing

% a = train(train(:,785)==4,:);
% figure;
% colormap('gray');
% plotsize = ceil(20);
% for i = 1:20
%   centr= a(i,:);
%   subplot(plotsize,plotsize,i);
%     
%   imagesc(reshape(centr,[28 28])');
% 
% end
% First parameter is a dataset. This dataset should have no outliers. The
% second parameter is the maximum k that you want to test. The 3rd paramter
% is the number of iterations
distance = elbow_method(train, 20, 10);
figure()
plot(distance);

train_sum = sum(train)/1500;
figure;
plot(train_sum)

figure;
colormap('gray');
heatmap = reshape(train_sum(1,1:784), [28 28]);
imagesc(heatmap'); 


%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids
% There are other ways you might initialize centroids.
% ***Feel free to experiment.***
% Note that this function takes two inputs and emits one output (y).


    
function y=initialize_centroids(data,num_centroids)
    centroid_type = "custom_generated_10";

    if centroid_type == "default"
        random_index=randperm(size(data,1));

        centroids=data(random_index(1:num_centroids),:);

        y=centroids;

    elseif centroid_type == "custom_generated_10"
        imagefiles = dir('generated_centroids\*.bmp');      
        nfiles = length(imagefiles);    % Number of files found
        images = zeros(nfiles, 785); 
        for ii=1:nfiles
           currentfilename = imagefiles(ii).name;
           currentimage = imread(fullfile('generated_centroids',currentfilename));
           images(ii,:) = [reshape(currentimage.', 1, []) 0];
        end
        
        y = images;
    end
end
%% Function to pick the Closest Centroid using norm/distance
% This function takes two arguments, a vector and a set of centroids
% It returns the index of the assigned centroid and the distance between
% the vector and the assigned centroid.

% This function has bug. Index is not an integer but it has the same value
% as vector distance

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
    norms = zeros(size(centroids,1),1);
    for i = 1:size(centroids,1)
        norms(i) = norm(data - centroids(i,:))^2;
    end
    [vec_distance, index] = min(norms);
    return;
end


%% Function to compute new centroids using the mean of the vectors currently assigned to the centroid.
% This function takes the set of training images and the value of k.
% It returns a new set of centroids based on the current assignment of the
% training images.
function new_centroids=update_Centroids(data,K)

    new_centroids = zeros(K, 785);
         
    for i = 1:K
        
      centroid_vectors = data((data(:, 785)==i), 1:784); 
      new_centroids(i,:) = [mean(centroid_vectors) 0];  
    end
end

%% Function to determine the optimal number of clusters
% This function assumes that the data set is "perfect". This also means
% the dataset should not have any outliers. This method is also known as
% the elbow method. The idea is to iterate the k-means algorithm for
% certain number of times. Then it caluclates the sum-square-distance of
% each datapoints to all centroids. The idea of having optimal numbers of
% centroids is to have the minimal distance between a given centroids to
% its "assumed" datapoints. Therefore, the smaller the SSD for each case of
% a number of centroids, the better that decided number of centroids is
% Doucmentation: https://en.wikipedia.org/wiki/Elbow_method_(clustering)
function distance = elbow_method(train, k_max, max_iter)
    distance = zeros(k_max,2);
    for k=1:k_max
        training_size = size(train,1);
        centroids=initialize_centroids(train,k);
        distances_to_centroids = zeros(1500,2);
        for iter=1:max_iter
            %assinging centroids to all the vectors
            for i = 1:training_size
                [centroid_index, distance_to_centroids] = assign_vector_to_centroid(train(i,:),centroids);
                distances_to_centroids(i,2) = centroid_index;
                distances_to_centroids(i,1) = distance_to_centroids;
            end    
            centroids = update_Centroids(train, k);
        end
        num_assignments = unique(distances_to_centroids(:,2));
%         Unique() returns an sorted array of unique numbers within the parameter
%         set. Therefore, it is only necessary to take the last element
        max_assignments = num_assignments(end);
        SSD_of_a_centroid = 0;
%         This loop implements the Sum of Squared Distance algorithms. This
%         algorthim is brute force, so it will be impratical if there are a
%         lot of centroids.
        for i=1:max_assignments
            temp = distances_to_centroids(distances_to_centroids(:,2) == i);
            SSD_of_a_centroid = SSD_of_a_centroid + sum(temp(:,1));
        end
        distance(k,1) = k;
        distance(k,2) = SSD_of_a_centroid;
    end
    return;
end 
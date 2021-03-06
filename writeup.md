# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

The code for this step is contained in the [training_data.py] file. The
[load_training_data()] method loads the vechile and non-vehicle images,
calculate [HOG][calculate_hog_features] and [color][calculate_color_features]
features for each image and [associate each image with a label].
I used the `skimage.hog()` method to obtain HOG features for each channel of
a training image. I tried RGB and YUV color spaces, but YUV color space gave
higher accuracy of classification. I tried different parameters of HOG
features extraction (9 and 11), different size of a cell (8x8 and 16x16),
different size of a block (1x1, 2x2, 3x3). I settled on 11 orientations,
16x16 pixels per cell and 2x2 cells per block, because it gave the highest
accuracy of classification. I merge HOG features calculated for each of
the YUV channels of a training image. I [combine][combine_features] HOG
features with color features. To obtain color features I down scale image
to the size of 16x16 pixels and flatten the resulted image into a vector
of floats.

If you run the [training_data.py] script, it will extract features and
generate labels represented by numpy arrays and [save][save_training_data]
them into the `traingin_data.pkl` file. This will speed up loading of
the training data during iterative training of the classifier.

[training_data.py]: ./training_data.py
[load_training_data()]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/training_data.py#L11
[calculate_hog_features]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/training_data.py#L46
[calculate_color_features]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/training_data.py#L56
[associate each image with a label]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/training_data.py#L37
[save_training_data]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/training_data.py#L67
[combine_features]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/training_data.py#L44

#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the [classification.py] file. I
trained the multi-layer perceptron using the `sklearn.neural_network.MLPClassifier`
class. First, I [split training and test data] using the `sklearn.model_selection.train_test_split`
method. Then I [normalize training features] using the `sklearn.preprocessing.StandardScaler`
class. Then [train the classifier] and [save the trained model] into a file.
Loading the trained model from a file speeds up detection pipeline.

If you run the [classification.py] script, it will train the classifier and
save the trained model into the `classifier.pkl` file.

[classification.py]: ./classification.py
[split training and test data]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/classification.py#L14
[normalize training features]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/classification.py#L19
[train the classifier]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/classification.py#L24
[save the trained model]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/classification.py#L38

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I've implemented a multiscale sliding window search. The code for this step
is contained in the [pipeline.py] script, the [`Pipeline.sliding_windows()`]
method.
The size of window and number of scales is controlled by the global variable
[`WINDOW_SIZES`][window_sizes]. I've selected several small sizes to detect
the white car, when it's far (between 25 and 30 seconds of the project
video). Big sizes should catch nearby cars.
For each of the window scales
I scan 3 overlapped rows near the [`HORIZON_LINE = 440`][horizon_line]. I've
noticed that the horizon line cuts a car image of any scale with the
approximate ratio of 1:2 (one part above horizon line and two parts below
the horizon line), which is represented by variable. Based on this fact,
I [calculate the position of each row] of windows. Each row of windows
overlaps an adjacent row by 3/4 of the window size. Each window in a row
overlaps the adjacent window by 3/4 of the window size. I've started with
the amount of overlapping equal to 1/2 of the window size. But bigger
overlapping can produce a more contrast heat map, that's why I've started
increasing overlapping and stopped on the selected numbers, because further
increasing means more processing time. The `sliding_windows()` method yields
a list of index masks, that can be applied to the original image to
retrieve the corresponding windows.

[pipeline.py]: ./pipeline.py
[window_sizes]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L9
[`Pipeline.sliding_windows()`]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L18
[horizon_line]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L10
[calculate the position of each row]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L25

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize the performance of the classifier, I've tried the following
things:
* mix in color features with HOG features
* using different color spaces (RGB and YUV)
* different parameters of HOG extraction (number of orientations, number of
  cells, number of blocks)
* hard negative mining

Here are some example images of working classifier:

![test2](./output_images/test2.jpg)
![test2](./output_images/test3.jpg)
![test2](./output_images/test4.jpg)
![test2](./output_images/test6.jpg)

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.avi)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.
From the positive detections I [created a heatmap]
by [accumulating probabilities] of the positive detections inside the map over
the time and then [thresholded that map]
to identify vehicle positions. I then used `scipy.ndimage.measurements.label()`
to [identify individual blobs] in the heatmap. Then [assumed] each blob
corresponded to a vehicle. I [filtered out] those blobs that look tall.
I [constructed bounding boxes] to cover the area of each blob left.

[created a heatmap]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L34
[accumulating probabilities]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L41
[thresholded that map]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L48
[identify individual blobs]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L49
[assumed]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L62
[filtered out]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L66
[constructed bounding boxes]: https://github.com/mode89/CarND-Vehicle-Detection/blob/master/pipeline.py#L51

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue that I faced, was the presence of the false positives. As a
workaround I performed hard negative mining on the test image. It worked,
but there is still a chance that the classifier will fail on the new unseen
data. Using a bigger training dataset will make the algorithm more robust.

Another issue is the computational time - the algorithm cannot perform in
real time. Current implementation performs resizing and feature extraction
for each of the windows separately. Those windows overlap each other, which
means, if we could resize and extract features from the whole ROI, we can
avoid redundant operations and improve frame rate.

Yet another issue was the detection of the distant white car. Choosing a
smaller size of the searching window and accumulation of the heatmap over
the time solved the problem.

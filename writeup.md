## Writeup Template

---

**Vehicle Detection Project**

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
a training image. I used the default value of 9 for the number of
orientations and the default value of `(8, 8)` for the size of a cell. After
I settled with the classifier, I tried values of `(1, 1)` and `(3, 3)` for
the number of cells per a block and selected the value of `(1, 1)`, because
it provided slightly higher FPS and I didn't notice a big difference
in the quality of detection compare to the value of `(3, 3)`. I merge HOG
features calculated for each of the RGB channels of a training image. I
[combine][combine_features] HOG features with color features. To obtain
color features I down scale image to the size of 16x16 pixels and flatten
the resulted image into a vector of floats.

If you run the [training_data.py] script, it will extract features and
generate labels represented by numpy arrays and [save][save_training_data]
them into the `traingin_data.pkl` file. This will speed up loading of
the training data during iterative training of the classifier.

[calculate_hog_features]: https://github.com/mode89/CarND-Vehicle-Detection/blob/f2cecc3119066e074928f17c22c0d23bc73012ec/training_data.py#L45
[calculate_color_features]: https://github.com/mode89/CarND-Vehicle-Detection/blob/f2cecc3119066e074928f17c22c0d23bc73012ec/training_data.py#L55
[associate each image with a label]: https://github.com/mode89/CarND-Vehicle-Detection/blob/f2cecc3119066e074928f17c22c0d23bc73012ec/training_data.py#L37
[save_training_data]: https://github.com/mode89/CarND-Vehicle-Detection/blob/f2cecc3119066e074928f17c22c0d23bc73012ec/training_data.py#L66
[combine_features]: https://github.com/mode89/CarND-Vehicle-Detection/blob/f2cecc3119066e074928f17c22c0d23bc73012ec/training_data.py#L43

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


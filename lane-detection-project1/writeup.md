
# **Finding Lane Lines on the Road** 

## Goal: To identify lane lines on the road, first in an image, and later in a video stream (really just a series of images)

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

### 1. Pipeline Description

The pipeline consists of the following steps. 

-   Convert the image to grayscale using the helper function grayscale.
2. Apply guassian kernel of size 5X5 to smooth the image.
3. Use Canny edge detection to determine image edges. The lower and upper threshold was set at 50 and 150.
4. Define an empty mask. Depending on the image, fill the mask using the color channels and create a polygon region of interest in the image.
5. Transform the masked image to Hough space using the helper function hough_lines.
6. The lines returned by the hough helper function are then passed to the draw_lines function.
   I modified the draw_line function as follows:
   > 6.1. Classify the hough lines to left and right lines based on slope. ( < -0.5 or > 0.5. I tried with 0 but it was not working for some of the images.<br>
   > 6.2. Using the polyfit function, found the linear fit for these left and right coordinates and determine the slope and intercept of each fitted line.<br>
   > 6.3. Determing the y coordinates of the line to be drawn based on the image size(since we know the size of the images being used). Using the y coordinates and the slope and intercept from 6.2, determine the x coordinates of the lines to be drawn.<br>
   > 6.4. Draw the lines detected.<br>
7. Merge the lines drawn in the above function and the input image to get the output.

### 2. Potential shortcomings 

-   For the challenge video, the function fails to detect the lane correctly when there is a patch on the road.
2. The function works for the example images. If there are more images with different scenarios like shadow, lighting,    rain, pedestrians etc, it may not work.
3. The vertices are determined using the image size and with the added knowledge of approximately where in the image the lane is located. So, it is pretty much hard coded.

### 3. Improvements to the pipeline

-   Optimization of code.
2. Find better way to determine the region of interest, ie, more dynamically.
3. To be able to detect lanes even with patches and other obstructions on the road.


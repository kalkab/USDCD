# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

I have created a function called LaneFinder which processes the images. In this function I have on first place converted the images to grayscale. Once I have done that I have applied the optimal edge detector via the Canny algorithm. Then I have set the ‘region of interest’ in the images which was already provided in the help algorithms. Hough Line transformation has been applied in order to detect straight lines and respectfully the lane lines. 

As a last step I have modified the draw_lines function and currently the code is drawing them separately (first left than right line). What I have done on first place is to figured it out if I am looking at the left or right lane based on the slope of the provided line. I have append all the x and y components and I have fitted the data to a first order polynomial. Next step was to extrapolate the data based on the acquired polynomial. Lasts steps were to draw the lines via the opencv line function and all the points after the extrapolation. I have made a condition when drawing not to look at points in the range of 0:300 in y axis. 

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]

### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when the whole road is on constant turn (left or right). Currently I have applied first order polynomial. Perhaps 2nd  order will be better for that purposes but I will need to test that in various scenarios. 
Another shortcoming could be when we are driving uphill or downhill. This would certainly affect the current script as it is not so advanced. 
One more thing would be incorrectly estimation of the lanes when the road is curvy. That would affect the simple lane separation method that I have implemented and the slope would be varying all the time.  

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to improve the lane identification method (left or right line).
Another potential improvement could be to apply 2nd order polynomial and to create an adaptable script in order to know where the road starts and where is the sky, etc.


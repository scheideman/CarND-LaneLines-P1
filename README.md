#**Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Video showing pipeline in action: https://youtu.be/H0cUdBW49Ao

###Computer Vision Pipeline:

1. Convert image to HSV space and extract out Value channel, from my experiments the Value channel gave better
results than the greyscale image, as the Value channel was better for seeing the bright yellow line
2. Gaussian blur with kernel size of 5
3. Canny edge detection with a lower threshold of 50 and upper threshold of 150
4. Extract out region of interest starting 0.6 of the way down in the y direction. The region is roughly a triangle
with the bottom conners in the left and right bottom corners of the image and peaking to a flat top at x ~ 0.5
5. Apply hough transform with rho, theta, threshold, min_line_len, max_line_gap equal to 1, 1 degree, 50, 50, 90
6.Separate lines with negative slope to the left line and positive slope to the right line and exclude lines with
infinite slope (vertical lines) and lines that are less than 15 degrees from the horizontal plane.
7. Take the average of all the left lines and again for all the right lines to get the average m and b value for
both left and right lines
8. Mix calculated line with prior lane line values to prevent jerky movements
9. Use calculated m and b values to find lines start and end points going from largest y value to smallest y value

Although my pipeline works well on the videos provided for the project, there are some areas I think it could be improved.
First, to make this algorithm more robust I would look into adding a curved line model, since I think my current
algorithm would fail in situations where the lane line is extremeley curved like on some overpass exit ramps. I did
explore using numpy's polyfit function but could not make it work better than my current pipeline.Second, modifying
the region of interest selection to be more dynamic would help improve this algorithm, as the lane line are not always 
going to be within the region I hardcoded. Third, canny edge detection requires parameter tuning, as a result 
variable condition eg, rain, snow, lighting could mean it needs to be retuned, therefore using a parameterless
edge detection method would be beneficial in this situation. 


### Setting up Environment

**Step 1:** Getting setup with Python

To do this project, you will need Python 3 along with the numpy, matplotlib, and OpenCV libraries, as well as Jupyter Notebook installed. Recommended to use Anaconda environment

**Step 2:** Installing OpenCV

`> pip install pillow`  
`> conda install -c menpo opencv3=3.1.0`

**Step 3:** Installing moviepy  

`>pip install moviepy`  

**Step 4:** Opening the code in a Jupyter Notebook

`> jupyter notebook`

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math

left_lane_prior = [None,None]
right_lane_prior = [None,None]
points_left_prior = []
points_right_prior = []



def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def fit_curve(points, x_min, x_max):
    """
    Fit a curve using polyfit function
    """
    points = np.array(points)
    x = points[:,0]
    y = points[:,1]

    z = np.polyfit(x, y, 2)
    curve_model = np.poly1d(z)

    # calculate points using model
    x_new = np.linspace(x_min, x_max, 100)
    y_new = curve_model(x_new)

    points_new = np.int32([[x_new[i],y_new[i]] for i in range(len(x_new))])

    return points_new

def fit_average_line(lines, target_x, x_max, y_min, y_max, lane_prior):
    """
    Fit a straight line using the average of lines and mixing with lane_prior

    lane_prior is the prior lines intercept and slope
    """
    mix = 0.2
    lines = np.array(lines)

    if(len(lines) == 0):
        mix = 0
        b_avg = 0
        m_avg  = 0
    else:
        mix = 0.2
        b_list = lines[:,1]
        m_list = lines[:,0]
        b_avg = sum(b_list)/len(b_list)
        m_avg  = sum(m_list)/len(m_list)

    if( lane_prior[0] is not None or lane_prior[1] is not None):
        b_avg = b_avg * mix + lane_prior[1] * (1-mix)
        m_avg = m_avg * mix + lane_prior[0] * (1-mix)

    x_tmp = int(round((y_max-b_avg)/m_avg))
    y_tmp = int(round((m_avg*target_x + b_avg)))

    if(x_tmp < x_max and x_tmp > 0):
        line = [[x_tmp,y_max,int(round((y_min-b_avg)/m_avg)),round(y_min)]]
    else:
        line = [[target_x,y_tmp,int(round((y_min-b_avg)/m_avg)),round(y_min)]]

    return line, m_avg,b_avg


def draw_lines(img, lines, color=[255, 0, 0], thickness=13):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def draw_lane_lines_curved(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Finds and draw left and right lane lines on a image using curve model
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    imshape = line_img.shape
    b_left = []
    m_left = []
    b_right = []
    m_right = []

    points_left = []
    points_right = []

    count = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = ((y2-y1)/(x2-x1))
            b = y2 - m * x2

            if(np.isinf(m) or abs(math.atan(m)*180/np.pi) < 15):
                continue
            if(m < 0):
                points_left.append([x1,y1])
                points_left.append([x2,y2])
                b_left.append(b)
                m_left.append(m)
            else:
                points_right.append([x1,y1])
                points_right.append([x2,y2])
                b_right.append(b)
                m_right.append(m)

    #get left lane line
    if(len(b_left) > 0):
        linesMB = [[m_left[i],b_left[i]] for i in range(len(b_left))]
        left_line, m_avg, b_avg  = fit_average_line(linesMB,0,imshape[1], imshape[0]*0.6,imshape[0],left_lane_prior)
        points_left.append([left_line[0][0],left_line[0][1]])
        new_points = fit_curve(points_left,0,imshape[1]*0.48)
        cv2.polylines(line_img,[new_points],False,(255,0,255),15)
    #get right lane line
    if(len(b_right) > 0):
        linesMB = [[m_right[i],b_right[i]] for i in range(len(b_right))]
        right_line, m_avg, b_avg = fit_average_line(linesMB,imshape[1],imshape[1],imshape[0]*0.6,imshape[0],right_lane_prior)
        points_right.append([right_line[0][0],right_line[0][1]])
        draw_lines(line_img, [right_line])
        new_points = fit_curve(points_right,imshape[1]*0.55,imshape[1])
        cv2.polylines(line_img,[new_points],False,(255,0,255),15)

    return line_img


def draw_lane_lines_straight(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Finds and draw left and right lane lines on a image using straight line model
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    imshape = line_img.shape
    b_left = []
    m_left = []
    b_right = []
    m_right = []
    lane_lines = []

    count = 0
    if(lines is None):
        return line_img
    for line in lines:
        for x1,y1,x2,y2 in line:
            #find slope and intercept
            m = ((y2-y1)/(x2-x1))
            b = y2 - m * x2

            #exclude vertical lines (inf slope) and near horizontal lines
            if(np.isinf(m) or abs(math.atan(m)*180/np.pi) < 15):
                continue
            #sort based on slope value
            if(m < 0):
                b_left.append(b)
                m_left.append(m)
            else:
                b_right.append(b)
                m_right.append(m)

    #get left lane line
    if(len(b_left) > 0 or left_lane_prior[0] is not None):
        linesMB = [[m_left[i],b_left[i]] for i in range(len(b_left))]
        left_line, m_avg, b_avg = fit_average_line(linesMB,0,imshape[1], imshape[0]*0.6,imshape[0],left_lane_prior)
        lane_lines.append(left_line)
        left_lane_prior[0] = m_avg
        left_lane_prior[1] = b_avg
    #get right lane line
    if(len(b_right) > 0 or right_lane_prior[0] is not None):
        linesMB = [[m_right[i],b_right[i]] for i in range(len(b_right))]
        right_line, m_avg, b_avg = fit_average_line(linesMB,imshape[1],imshape[1],imshape[0]*0.6,imshape[0], right_lane_prior)
        lane_lines.append(right_line)
        right_lane_prior[0] = m_avg
        right_lane_prior[1] = b_avg

    draw_lines(line_img, lane_lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., lamda=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:


    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lamda)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    ysize = image.shape[0]
    himage = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    hsvs = cv2.split(himage)

    #Use value channel from HSV to better see yellow lane lines
    img = hsvs[2]
    imshape = image.shape

    img = gaussian_blur(img, 5)
    img = canny(img,50,150)
    vertices = np.array([[(0,imshape[0]),(imshape[1]*0.48, imshape[0]*0.6), (imshape[1]*0.52, imshape[0]*0.6), (imshape[1],imshape[0])]], dtype=np.int32)
    img = region_of_interest(img, vertices)

    #Function for drawing lanes lines using Hough Transform
    img = draw_lane_lines_straight(img,1, np.pi * 1/180, 50, 50, 90)

    result = weighted_img(img, image, alpha=0.8, beta=0.9, lamda=0.0)

    return result


def main():
    cap = cv2.VideoCapture('udc_sdc2.avi')

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = process_image(frame)
        plt.imshow(img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()

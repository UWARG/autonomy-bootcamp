#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

// The main function will read an image, convert it from BGR to HSV, and then threshold the red regions in an image.
// It'll first create two images: one image thresholding regions falling between the lower red Hue value (that is 0-10),
// and the other image thresholding regions in the range of the upper red Hue values (160-179)
// Both images are combined into one, single image by using addWeighted in the OpenCV library.
// The output is then displayed onto the screen.

// improvements to make: remove noise, add a corner detector.

int main( int argc, char** argv )
{
    if( argc != 2)
    {
        cout <<" To run in terminal: $ ./opencvtst <yourimagename> " << endl;
        return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    // Reads the image provided in terminal.

    if(image.empty())
    {
        cout <<  "Image not found. Make sure it is near your program file." << endl ;
        return -1;
    }

    // Checks that a valid image was given.

    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // converts the BGR image to HSV (Hue Saturation Value) image

    Mat lowerRedHueRange;
    inRange(hsvImage, Scalar(0, 100, 100), Scalar(10, 255, 255), lowerRedHueRange);

    Mat upperRedHueRange;
    inRange(hsvImage, Scalar(160, 100, 100), Scalar(179, 255, 255), upperRedHueRange);

    // Red has Hue values of 0-10 and 160-180 in OpenCV

    Mat redImage;
    addWeighted(lowerRedHueRange, 1.0, upperRedHueRange, 1.0, 0.0, redImage);

    // combines the images within the lower red Hue range and the upper red Hue range.


    namedWindow( "Thresholded Image", CV_WINDOW_AUTOSIZE);

    // Create a window for display.

    imshow("Red Threshold", redImage);

    // displays the segmented red image

    waitKey(0);

    // Wait until keystroke.

    return 0;
}


cd Project_2D_Feature_Matching/results
convert -delay 100 -loop 0 ./HARRIS_BRIEF/*.png ./HARRIS_BRIEF/result.gif
convert -delay 100 -loop 0 ./SHITOMASI_BRIEF/*.png ./SHITOMASI_BRIEF/result.gif
convert -delay 100 -loop 0 ./ORB_BRIEF/*.png ./ORB_BRIEF/result.gif
convert -delay 100 -loop 0 ./SIFT_BRIEF/*.png ./SIFT_BRIEF/result.gif
convert -delay 100 -loop 0 ./AKAZE_BRIEF/*.png ./AKAZE_BRIEF/result.gif
convert -delay 100 -loop 0 ./BRISK_BRIEF/*.png ./BRISK_BRIEF/result.gif
convert -delay 100 -loop 0 ./FAST_BRIEF/*.png ./FAST_BRIEF/result.gif

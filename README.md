# Harris-Corner-Detection-and-its-variants
Implementation of Harris Corner Detection and its variants

Algorithm:
1. For a given image, compute the second moment matrix M at each pixel

2
2
( , ) x x y
x y y
I I I
w x y
I I I
 
=      
M 

, where
I
x x
I


= ,
I
y y
I


= , and w(x, y) is a window function.

Use following window functions
i) Gaussian (with  = 1, 1.5) function
ii) Box function of same size as it used in (i)
2. Compute cornerness C using the formulas given by
i) Harris & Stephens (1988),

2 C M M = − det( ) trace ( ) 

ii) Kanade & Tomasi (1994), C = min( , )  1 2

, where s are Eigen values of M

iii) Nobel (1998),

det( )
trace( )
M C M +
=

3. Chose appropriate threshold on C to pick high cornerness
4. Apply Non-maxima suppression in a 3×3 neighborhood to pick the peaks
Additional Info
• Write a function, myImgConvolve( Input_image, kernel), that convolves an image with a
given convolution kernel. Your function should output image of the same size as that of
input Image (use necessary padding).
• Use myImgConvolve function to get image derivatives
o Generate Gaussian derivative Kernel (σ =1, 2). Use at least (3σ +1)×(3σ +1) as the
size of Kernel (You may write a separate function to generate the kernels.)
• All the output images for various combination of parameters at different stage of the
program should be saved with proper representation and filename

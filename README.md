# Project_image-processing
Project Objectives：There are some photovoltaic panels that need to be checked by the staff to see whether they work properly.If the photovoltaic panels didn't work, some electronic components went wrong in the photovoltaic panels. But the electronic components must be nchecked by the staff inside one by one . So they can't be shoot by ordinary camera directly. How to know the status of electronic components in the photovoltaic panel by just taking a few pictures automatically.
Solution:By using drone with two cameras,we can capture the heat and appearance of electronic components respectively inside the photovoltaic panel.These two cameras are infrared camera which can capture the heat of electronic components inside and natural light camera which can appearance the heat of photovoltaic panel. We can use infrared camera to catch the heat state of the components' heat is abnormal if they are damaged. So we can use image processing to find out the location of damaged electronic components. Then put them on the picture photoed by natural light camera, we can know which part of the photovoltaic panel is damaged.
Method：I put solution into two parts. 
The first part is to find out the damaged components.
1、input and read picture：You must put the picture under the contents and put their initial name and format in the same.
2、I turn the image into a grayscale image. By using k-means algorithm, picture can be divided into different parts.
3、I connect image edges by close operation.Then using canny factor extracts pictures' edge.
4、Then I use close operation connects scatter edge，fill the edges of the connection,open operation to remove scatter.
The second part is to Stitch image.
I find the best match by finding the feature points and correct the distortion image.

Step
1、You can input the initial name of picutre and format (jpg:j,bmp:b,png:p)
2、Input the k-means number.
3、Input Canny factor
4、Click and select four corners of the image

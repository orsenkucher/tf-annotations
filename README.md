# Image Object Detection Program
This program traverses a directory of image files, extracts their dimensions, and exports the data in CSV format for use in object detection machine learning models.

## Requirements
This program requires the Rust programming language to be installed on your system. You can download and install Rust from the official website: https://www.rust-lang.org/tools/install.

## Usage
To use this program, run the following command in your terminal:

```
cargo run --release -- <directory_path>
```
Replace <directory_path> with the path to the directory containing the image files you want to process.

The program will traverse the directory, extract the dimensions of each image file, and export the data in CSV format to a file named tensorflow.csv in the root directory of the program.

## Supported Image Formats
This program supports the following image formats:

* JPEG/JPG
* PNG
* BMP

Any image files with extensions other than these formats will be ignored.

## Output Format
The exported CSV file has the following format:

```csv
filename,width,height,class,xmin,ymin,xmax,ymax
```
* `filename`: the name of the image file
* `width`: the width of the image in pixels
* `height`: the height of the image in pixels
* `class`: the name of the class (i.e. directory name) containing the image file
* `xmin`: the x-coordinate of the top-left corner of the bounding box (set to 0 for this program)
* `ymin`: the y-coordinate of the top-left corner of the bounding box (set to 0 for this program)
* `xmax`: the x-coordinate of the bottom-right corner of the bounding box (set to the width of the image for this program)
* `ymax`: the y-coordinate of the bottom-right corner of the bounding box (set to the height of the image for this program)

## License
This program is licensed under the MIT License. See the LICENSE file for more information.

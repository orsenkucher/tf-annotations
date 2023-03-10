use std::path::Path;

use anyhow::Result;
use image::GenericImageView;

struct ObjectDetection {
    // Path from root directory to the image.
    filename: String,
    width: u32,
    height: u32,
    // Directory name containing image.
    class: String,
}

fn main() -> Result<()> {
    // Get the path to the image file from the command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];

    // Call the directory traversal function
    let result: Vec<ObjectDetection> = traverse_images(dir)?;

    Ok(())
}

fn traverse_images<P: AsRef<Path>>(root: P) -> Result<Vec<ObjectDetection>> {
    // TODO: traverse directories in `root` directory.
    // Each directory contains images, use `image_size` function to get their dimensions.
    Ok(vec![])
}

fn export(data: Vec<ObjectDetection>) -> Result<()> {
    // TODO: export to `tensorflow.csv` file in format:
    // filename,width,height,class,xmin,ymin,xmax,ymax
    // xmin,ymin are zero; xmax,ymax equal to image width and height respectively.
    Ok(())
}

fn image_size<P: AsRef<Path>>(path: P) -> (u32, u32) {
    // Open the image file and decode it using the image crate
    let img = image::open(path).unwrap();

    // Get the dimensions of the image and return them
    img.dimensions()
}

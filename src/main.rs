use std::path::Path;

use image::GenericImageView;

fn main() {
    // Get the path to the image file from the command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let filename = &args[1];

    // Call the image_size function with the file path
    let (width, height) = image_size(filename);

    // Print the size of the image
    println!("Image size: {} x {}", width, height);
}

fn image_size<P: AsRef<Path>>(path: P) -> (u32, u32) {
    // Open the image file and decode it using the image crate
    let img = image::open(path).unwrap();

    // Get the dimensions of the image and return them
    img.dimensions()
}

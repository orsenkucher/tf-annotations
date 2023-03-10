use std::fs;
use std::path::Path;

use anyhow::{anyhow, Result};
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

    // Export the result to a CSV file
    export(result)?;

    Ok(())
}

fn traverse_images<P: AsRef<Path>>(root: P) -> Result<Vec<ObjectDetection>> {
    // Initialize an empty vector to hold the image objects
    let mut objects = Vec::new();

    // Traverse the root directory and its subdirectories
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();

        // If the path is a directory, recursively traverse its contents
        if path.is_dir() {
            let class = path
                .file_name()
                .ok_or_else(|| anyhow!("Invalid path"))?
                .to_string_lossy()
                .to_string();

            for image_entry in fs::read_dir(path)? {
                let image_entry = image_entry?;
                let image_path = image_entry.path();

                // If the path is an image file, get its dimensions and add it to the list of objects
                if let Some(extension) = image_path.extension() {
                    if extension == "jpg" || extension == "jpeg" || extension == "png" {
                        let (width, height) = image_size(&image_path);
                        let object = ObjectDetection {
                            filename: image_path.to_string_lossy().to_string(),
                            width,
                            height,
                            class: class.clone(),
                        };
                        objects.push(object);
                    }
                }
            }
        }
    }

    Ok(objects)
}

fn export(data: Vec<ObjectDetection>) -> Result<()> {
    // Open the CSV file for writing
    let mut writer = csv::Writer::from_path("tensorflow.csv")?;

    // Write the header row
    writer.write_record(&[
        "filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax",
    ])?;

    // Write the data rows
    for object in data {
        writer.write_record(&[
            &object.filename,
            &object.width.to_string(),
            &object.height.to_string(),
            &object.class,
            "0",
            "0",
            &object.width.to_string(),
            &object.height.to_string(),
        ])?;
    }

    Ok(())
}

fn image_size<P: AsRef<Path>>(path: P) -> (u32, u32) {
    // Open the image file and decode it using the image crate
    let img = image::open(path).unwrap();

    // Get the dimensions of the image and return them
    img.dimensions()
}

use std::fs;
use std::path::Path;

use anyhow::{anyhow, Result};
use image::GenericImageView;
use rayon::prelude::*;

struct ObjectDetection {
    // Path from root directory to the image
    filename: String,
    width: u32,
    height: u32,
    // Directory name containing image
    class: String,
}

fn main() -> Result<()> {
    // Get the path to the image file from the command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];

    // Call the directory traversal function
    let result = traverse_images(dir)?;

    // Export the result to a CSV file
    export(result)?;

    Ok(())
}

fn traverse_images<P: AsRef<Path>>(root: P) -> Result<Vec<ObjectDetection>> {
    // Get a list of all directories in the root directory
    let dirs: Vec<_> = fs::read_dir(root)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .collect();

    // Parallelize the processing of directories using rayon
    let result: Vec<ObjectDetection> = dirs
        .into_par_iter()
        .flat_map(|entry| {
            // Get a list of all image files in the directory
            let files = fs::read_dir(entry.path())
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.path().is_file() && is_supported_image(&entry.path()));

            // Process the image files in the directory
            let detections = files
                .map(|entry| {
                    let path = entry.path();
                    let filename = path.file_name().unwrap().to_str().unwrap().to_string();
                    let class = entry
                        .path()
                        .parent()
                        .unwrap()
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string();
                    let (width, height) = image_size(&path);
                    ObjectDetection {
                        filename,
                        width,
                        height,
                        class,
                    }
                })
                .collect::<Vec<_>>();

            detections.into_par_iter()
        })
        .collect();

    Ok(result)
}

fn is_supported_image<P: AsRef<Path>>(path: P) -> bool {
    match path.as_ref().extension() {
        Some(ext) => {
            let ext_str = ext.to_string_lossy().to_lowercase();
            ext_str == "jpg" || ext_str == "jpeg" || ext_str == "png" || ext_str == "bmp"
        }
        None => false,
    }
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

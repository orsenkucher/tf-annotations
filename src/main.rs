use std::fs::{self, DirEntry};
use std::path::Path;

use anyhow::anyhow;
use anyhow::Result;
use image::GenericImageView;
use iter_tools::Itertools;
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
    let dir = args.get(1).map_or("images", |s| s);

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
    dirs.into_par_iter()
        .map(|entry| {
            // Get a list of all image files in the directory
            let files = fs::read_dir(entry.path())?
                .filter_map(Result::ok)
                .filter(|entry| entry.path().is_file() && is_supported_image(entry.path()));

            // Process the image files in the directory
            let detections = files
                .map(|entry| {
                    let path = entry.path();
                    let filename = path.file_name().unwrap().to_str().unwrap().to_string();
                    let class = extract_class(entry).ok_or_else(|| anyhow!("Class not found"))?;
                    let (width, height) = image_size(&path);
                    Ok(ObjectDetection {
                        filename,
                        width,
                        height,
                        class,
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(detections)
        })
        .flat_map_iter(|result| std::iter::once(result).flatten_ok())
        .collect()
}

fn extract_class(entry: DirEntry) -> Option<String> {
    Some(entry.path().parent()?.file_name()?.to_str()?.to_owned())
}

const SUPPORTED_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp"];

fn is_supported_image<P: AsRef<Path>>(path: P) -> bool {
    match path.as_ref().extension() {
        Some(ext) => {
            let ext_str = ext.to_string_lossy().to_lowercase();
            SUPPORTED_EXTENSIONS.contains(&ext_str.as_str())
        }
        None => false,
    }
}

fn export(data: Vec<ObjectDetection>) -> Result<()> {
    // Open the CSV file for writing
    let mut writer = csv::Writer::from_path("tensorflow.csv")?;

    // Write the header row
    writer.write_record([
        "filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax",
    ])?;

    // Write the data rows
    for object in data {
        writer.write_record([
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

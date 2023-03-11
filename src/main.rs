use std::fs::{self, DirEntry};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use image::GenericImageView;
use rayon::prelude::*;

#[derive(Debug)]
struct ObjectDetection {
    // Path from root directory to the image
    filename: PathBuf,
    width: u32,
    height: u32,
    // Directory name containing image
    class: String,
}

fn main() -> Result<()> {
    // Get the path to the image file from the command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map_or("images", |s| s);

    // Measure the execution time of the directory traversal
    let start_time = Instant::now();
    let result = traverse_images(dir)?;
    let elapsed_time = start_time.elapsed();

    // Export the result to a CSV file
    let export_start_time = Instant::now();
    export(result)?;
    let export_elapsed_time = export_start_time.elapsed();

    // Print the execution times
    println!(
        "Traversal time: {}.{:03}s",
        elapsed_time.as_secs(),
        elapsed_time.subsec_millis()
    );
    println!(
        "Export time: {}.{:03}s",
        export_elapsed_time.as_secs(),
        export_elapsed_time.subsec_millis()
    );

    Ok(())
}

fn traverse_images<P: AsRef<Path>>(root: P) -> Result<Vec<ObjectDetection>> {
    // Get a list of all directories in the root directory
    let dirs: Vec<_> = fs::read_dir(root)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .collect();

    // Parallelize the processing of directories using rayon
    let detections: Vec<Vec<_>> = dirs
        .into_par_iter()
        .map(|entry| {
            // Get a list of all image files in the directory
            let files = fs::read_dir(entry.path())?
                .filter_map(Result::ok)
                .filter(|entry| entry.path().is_file() && is_supported_image(entry.path()));

            // Process the image files in the directory
            let detections = files
                .map(|entry| {
                    let path = entry.path();
                    let class = extract_class(&entry).ok_or_else(|| anyhow!("Class not found"))?;
                    let (width, height) = image_size(&path)?;
                    Ok(ObjectDetection {
                        filename: path,
                        width,
                        height,
                        class,
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(detections)
        })
        .collect::<Result<_>>()?;

    // Flatten the nested vector of detections into a single vector
    let detections = detections.into_iter().flatten().collect::<Vec<_>>();

    Ok(detections)
}

fn extract_class(entry: &DirEntry) -> Option<String> {
    entry
        .path()
        .parent()?
        .file_name()?
        .to_str()?
        .to_owned()
        .into()
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

fn get_base_dir(filename: &PathBuf) -> &Path {
    Path::new(filename)
        .ancestors()
        .nth(1)
        .unwrap_or(Path::new(""))
}

fn get_relative_filename<'a>(filename: &'a PathBuf, base_dir: &Path) -> &'a Path {
    Path::new(filename)
        .strip_prefix(base_dir)
        .unwrap_or(Path::new(filename))
}

fn export(data: Vec<ObjectDetection>) -> Result<()> {
    // Open the CSV file for writing
    let mut writer = csv::Writer::from_path("tensorflow.csv")?;

    // Write the header row
    writer.write_record([
        "filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax",
    ])?;

    // Write the data rows
    for object in &data {
        let base_dir = get_base_dir(&object.filename);
        let filename = get_relative_filename(&object.filename, base_dir);
        let (xmin, ymin, xmax, ymax) = calculate_bounding_box(object.width, object.height);
        writer.write_record([
            &filename.to_string_lossy().to_string(),
            &object.width.to_string(),
            &object.height.to_string(),
            &object.class,
            &xmin.to_string(),
            &ymin.to_string(),
            &xmax.to_string(),
            &ymax.to_string(),
        ])?;
    }

    Ok(())
}

fn calculate_bounding_box(width: u32, height: u32) -> (u32, u32, u32, u32) {
    let x_center = width / 2;
    let y_center = height / 2;
    let w = (width as f32 * 0.8) as u32;
    let h = (height as f32 * 0.8) as u32;
    let xmin = x_center - (w / 2);
    let ymin = y_center - (h / 2);
    let xmax = xmin + w;
    let ymax = ymin + h;
    (xmin, ymin, xmax, ymax)
}

fn image_size<P: AsRef<Path>>(path: P) -> Result<(u32, u32), image::ImageError> {
    // Open the image file and decode it using the image crate
    let img = image::open(path)?;

    // Get the dimensions of the image and return them
    Ok(img.dimensions())
}

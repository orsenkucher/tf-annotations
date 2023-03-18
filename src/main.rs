use std::fs::DirEntry;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use image::GenericImageView;
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use structopt::StructOpt;

const IMAGES_DIR: &str = "images";
const CSV_FILE: &str = "tensorflow.csv";

/// A struct representing the label classification configuration, containing a vector of LabelGroup.
#[derive(Debug, Serialize, Deserialize)]
struct LabelClassification {
    groups: Vec<LabelGroup>,
}

/// LabelGroup: A struct representing a group of labels, containing the class
/// name and a vector of label strings.
#[derive(Debug, Serialize, Deserialize)]
struct LabelGroup {
    class: String,
    description: String,
    labels: Vec<String>,
}

#[derive(Debug, StructOpt)]
struct Cli {
    #[structopt(short, long, default_value = IMAGES_DIR)]
    dir: String,
}

/// A struct representing an object detection, containing the image file path,
/// dimensions, and class.
#[derive(Debug)]
struct ObjectDetection {
    filename: PathBuf,
    width: u32,
    height: u32,
    class: String,
}

/// The entry point of the program, responsible for reading the configuration,
/// traversing the images, exporting results, and printing execution times.
fn main() -> Result<()> {
    let args = Cli::from_args();
    let dir = &args.dir;

    let toml_str = include_str!("../label_classification2.toml");
    let classifier: LabelClassification = toml::from_str(toml_str)?;

    let start_time = Instant::now();
    let result = traverse_images(dir)?;
    let elapsed_time = start_time.elapsed();

    let export_start_time = Instant::now();
    export(&result, &classifier)?;
    let export_elapsed_time = export_start_time.elapsed();

    print_unique_classes(&result);

    println!("Traversal time: {}ms", elapsed_time.as_millis());
    println!("Export time: {}ms", export_elapsed_time.as_millis());

    Ok(())
}

/// Traverses the images in the given directory, returning a vector of
/// ObjectDetection instances.
fn traverse_images(dir: &str) -> Result<Vec<ObjectDetection>> {
    let dirs: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .collect();

    let detections: Vec<Vec<_>> = dirs
        .into_par_iter()
        .map(|entry| {
            let files = std::fs::read_dir(entry.path())?
                .filter_map(Result::ok)
                .filter(|entry| is_supported_image(entry.path()));

            let detections = files
                .map(|entry| {
                    let path = entry.path();
                    let class = extract_class(&entry)
                        .ok_or_else(|| anyhow!("Class not found for {:?}", entry.path()))?;
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

    Ok(detections.into_iter().flatten().collect())
}

/// Extracts the class name from the given directory entry.
fn extract_class(entry: &DirEntry) -> Option<String> {
    entry
        .path()
        .parent()?
        .file_name()?
        .to_str()
        .map(str::to_owned)
}

const SUPPORTED_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp"];

/// Checks if the given path points to a supported image file format.
fn is_supported_image<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_lowercase)
        .map(|ext_str| SUPPORTED_EXTENSIONS.contains(&ext_str.as_str()))
        .unwrap_or(false)
}

/// Returns the base directory of the given path.
fn get_base_dir(filename: &Path) -> &Path {
    filename.ancestors().nth(1).unwrap_or(Path::new(""))
}

/// Returns the relative filename of the given path relative to the base directory.
fn get_relative_filename<'a>(filename: &'a PathBuf, base_dir: &Path) -> &'a Path {
    filename
        .strip_prefix(base_dir)
        .unwrap_or(Path::new(filename))
}

/// Prints the unique classes found in the given vector of ObjectDetection instances.
fn print_unique_classes(data: &[ObjectDetection]) {
    let classes: Vec<_> = data
        .iter()
        .map(|detection| &detection.class)
        .unique()
        .collect();
    println!("Unique classes: {:?}", classes);
    println!("Unique len: {:?}", classes.len());
}

impl LabelClassification {
    /// Returns the class name associated with the given label
    /// from the LabelClassification instance.
    fn get_class(&self, label: &str) -> Option<&str> {
        self.groups
            .iter()
            .find(|group| group.labels.contains(&label.to_owned()))
            .map(|group| group.class.as_str())
    }
}

/// Exports the object detection data to a CSV file, using the given
/// LabelClassification instance for label classification.
fn export(data: &[ObjectDetection], classifier: &LabelClassification) -> Result<()> {
    let mut writer = csv::Writer::from_path(CSV_FILE)?;

    writer.write_record([
        "filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax",
    ])?;

    for object in data {
        let base_dir = get_base_dir(&object.filename);
        let filename = get_relative_filename(&object.filename, base_dir);
        let (xmin, ymin, xmax, ymax) = calculate_bounding_box::<80>(object.width, object.height);

        if let Some(class) = classifier.get_class(&object.class) {
            writer.write_record([
                &filename.to_string_lossy().to_string(),
                &object.width.to_string(),
                &object.height.to_string(),
                &class.to_owned(),
                &xmin.to_string(),
                &ymin.to_string(),
                &xmax.to_string(),
                &ymax.to_string(),
            ])?;
        } else {
            println!("Skipping: {}", &object.class)
        }
    }

    Ok(())
}

/// Calculates the bounding box for the given image dimensions and percentage.
fn calculate_bounding_box<const PERCENT: u32>(width: u32, height: u32) -> (u32, u32, u32, u32) {
    let x_center = width / 2;
    let y_center = height / 2;
    let w = (width as f32 * (PERCENT as f32 / 100.0)) as u32;
    let h = (height as f32 * (PERCENT as f32 / 100.0)) as u32;
    let xmin = x_center - (w / 2);
    let ymin = y_center - (h / 2);
    let xmax = xmin + w;
    let ymax = ymin + h;
    (xmin, ymin, xmax, ymax)
}

/// Returns the dimensions (width and height) of the given image file.
fn image_size(path: &Path) -> Result<(u32, u32), image::ImageError> {
    let img = image::open(path)?;
    Ok(img.dimensions())
}

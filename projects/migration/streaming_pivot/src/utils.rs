use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

pub fn get_buffered_reader(filename: &Path) -> io::BufReader<File> {
    let file = File::open(filename).expect("failed to open file");
    io::BufReader::new(file)
}
pub fn get_buffered_writer(filename: &Path) -> io::BufWriter<File> {
    let file = File::create_new(filename).expect("failed to open file");
    io::BufWriter::new(file)
}
pub fn get_column_names(filename: &Path) -> Vec<String> {
    let file = File::open(filename).expect("falied to open column names file");
    let reader = io::BufReader::new(file);

    let mut lines: Vec<String> = Vec::new();

    for line in reader.lines() {
        lines.push(line.expect("failed to read line"));
    }

    lines
}

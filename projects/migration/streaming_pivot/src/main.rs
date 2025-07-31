use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::usize;

const COL_NAMES: [&str; 38] = [
    "ai0879a",
    "ai0879b",
    "ai0879c",
    "ai0897b",
    "aic3730_mode",
    "aic3730_out",
    "aic3730_pv",
    "aic3730_sp",
    "aic3730_sprl",
    "aic3731_mode",
    "aic3731_out",
    "aic3731_pv",
    "aic3731_sp",
    "aic3731_sprl",
    "arc0879c_out",
    "fi_0871",
    "fi0872",
    "fic3734_mode",
    "fic3734_out",
    "fic3734_pv",
    "fic3734_sp",
    "fic3734_sprl",
    "fv3735_pv",
    "li3734",
    "m3730a_pv",
    "m3730b_pv",
    "m3731a_pv",
    "m3731b_pv",
    "m3739_pv",
    "pdic3738_mode",
    "pdic3738_out",
    "pdic3738_pv",
    "pdic3738_sp",
    "pdic3738_sprl",
    "pi0169",
    "rlcore_opcua_wd",
    "ti0880",
    "watchdog",
];

fn main() {
    let col_idx = COL_NAMES
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect::<HashMap<&str, usize>>();
    let long_tsv = Path::new("/home/kerrick/epcor_scrubber/data_copy-07_21_2025.txt");
    let wide_tsv = Path::new("/home/kerrick/epcor_scrubber/data_copy-07_21_2025_wide.txt");

    let buf_reader = get_buffered_reader(long_tsv);
    let mut buf_writer = get_buffered_writer(wide_tsv);

    let mut last_timestamp = String::from("");
    let mut line_data: Vec<String> = vec![String::from("\\N"); COL_NAMES.len()];

    for (i, line) in buf_reader.lines().map_while(Result::ok).enumerate() {
        let mut vals = line.split("\t");

        // Get timestamp and check if it is new
        let timestamp = vals.next().unwrap();
        if i == 0 {
            // i hope this check is compiled out
            last_timestamp = timestamp.to_string();
        }
        if timestamp != last_timestamp {
            // if current line has new timestamp,
            // write line data (built from previous lines) to file
            let new_line = last_timestamp + "\t" + &line_data.join("\t") + "\n";
            buf_writer.write(&new_line.as_bytes()).unwrap();

            // update last timestamp and reset line data
            last_timestamp = timestamp.to_string();
            line_data = vec![String::from("\\N"); COL_NAMES.len()];
        }

        // get columns we care about
        let name = vals.next().unwrap(); // second col
        let val = vals.next().unwrap(); // third col

        // combine process and name into single col
        let tag = name.to_lowercase();

        // add data to line
        let idx = col_idx[tag.as_str()];
        line_data[idx] = val.to_string();
    }
    // write last line
    let new_line = last_timestamp + "\t" + &line_data.join("\t") + "\n";
    buf_writer.write(&new_line.as_bytes()).unwrap();
    buf_writer.flush().unwrap();
}

fn get_buffered_reader(filename: &Path) -> io::BufReader<File> {
    let file = File::open(filename).expect("failed to open file");
    io::BufReader::new(file)
}
fn get_buffered_writer(filename: &Path) -> io::BufWriter<File> {
    //let file = File::open(filename).expect("failed to open file");
    let file = File::create_new(filename).expect("failed to open file");
    io::BufWriter::new(file)
}

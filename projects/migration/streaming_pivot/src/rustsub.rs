use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::usize;
use clap::Parser;
use regex::Regex;

mod utils;
use utils::{get_buffered_reader, get_buffered_writer, get_column_names};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    input_file: String,

    #[arg(short, long)]
    col_names: String,

    #[arg(short, long)]
    output_file: String,
}

// fn parse_row(line: &String) -> String {
//
//     // first, combine info in id, process, name 
//     // to construct tag name for PLC tags
//     // for some reason, we shouldn't escape
//     // the closing bracket in [^]] (any character except closing bracket)
//     // original: s/^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+00\t)ns=1;s=\[.+-([^]]+)\][^\t]+\t([^\t]+)\t([^\t]+)\t/\1\L\2_\3_\4\t/
//     let re1 = Regex::new(r"^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+00\t)ns=1;s=\[.+-([^]]+)\][^\t]+\t([^\t]+)\t([^\t]+)\t").unwrap();
//     // NOTE: referencing a capture group with $ followed by underscore results in a bug that is
//     // known and wont be fixed: https://github.com/rust-lang/regex/issues/69
//     // Need to workaround by using a separator "|" and then removing it.
//     let result1 = re1.replace(line, "$1$2|_$3|_$4\t").replace("|", "");
//
//     // add rlcore label to workaround tags from custom micro OPC
//     // and construct tag name
//     // original: s/^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+00\t)ns=1;[^\t]+\t([^\t]+)\t([^\t]+)\t/\1\Lrlcore_\2_\3\t/
//     let re2 = Regex::new(r"^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+00\t)ns=1;[^\t]+\t([^\t]+)\t([^\t]+)\t").unwrap();
//     let result2 = re2.replace(&result1, "$1rlcore_$2|_$3$t").replace("|", "");
//
//     // # clean up double uf1_uf1 or uf2_uf2
//     // # s/uf([12])_uf\1/uf\1/
//     // s/^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+00\t.+)uf([12])_uf\2/\1uf\2/
//     let re3 = Regex::new(r"^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+00\t.+)uf([12])_uf[12]").unwrap();
//     let result3 = re3.replace(&result2, "$1uf$2").to_lowercase();
//
//     result3
// }

fn main() {
    let args = Args::parse();
    println!("Input: {}", args.input_file);
    println!("Output: {}", args.output_file);
    println!("Col name: {}", args.col_names);

    let col_names_path = Path::new(&args.col_names);
    let col_names = get_column_names(col_names_path);

    let col_idx = col_names
        .clone()
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect::<HashMap<String, usize>>();

    // let long_tsv = Path::new("/home/kerrick/epcor_scrubber/data_copy-07_21_2025.txt");
    // let wide_tsv = Path::new("/home/kerrick/epcor_scrubber/data_copy-07_21_2025_wide.txt");

    let long_tsv = Path::new(&args.input_file);
    let wide_tsv = Path::new(&args.output_file);

    let buf_reader = get_buffered_reader(long_tsv);
    let mut buf_writer = get_buffered_writer(wide_tsv);

    // Add column names to Rust
    let header_line = String::from("time\t") + &col_names.join("\t") + "\n";
    buf_writer.write(&header_line.as_bytes()).unwrap();

    let mut last_timestamp = String::from("");
    let mut line_data: Vec<String> = vec![String::from("\\N"); col_names.len()];


    let re1 = Regex::new(r"^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+00\t)ns=1;s=\[.+-([^]]+)\][^\t]+\t([^\t]+)\t([^\t]+)\t").unwrap();
    let re2 = Regex::new(r"^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+00\t)ns=1;[^\t]+\t([^\t]+)\t([^\t]+)\t").unwrap();
    let re3 = Regex::new(r"^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\+00\t.+)uf([12])_uf[12]").unwrap();

    for (i, raw_row) in buf_reader.lines().map_while(Result::ok).enumerate() {

        // let row = parse_row(&raw_row);
        let row = re3.replace(&re2.replace(&re1.replace(&raw_row, "$1$2|_$3|_$4\t"), "$1rlcore_$2|_$3$t"), "$1uf$2").replace("|", "");
        let mut vals = row.split("\t");

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
            line_data = vec![String::from("\\N"); col_names.len()];
        }

        // get columns we care about
        let name = vals.next().unwrap(); // second col
        let val = vals.next().unwrap(); // third col

        // combine process and name into single col
        let tag = name.to_lowercase();

        // add data to line
        let idx = col_idx[tag.as_str()];
        line_data[idx] = val.to_string();
        //
        // // log
        // if i % 100_000 == 0 {
        //     println!("processing row {i}");
        // }
    }
    // write last line
    let new_line = last_timestamp + "\t" + &line_data.join("\t") + "\n";
    buf_writer.write(&new_line.as_bytes()).unwrap();
    buf_writer.flush().unwrap();
}

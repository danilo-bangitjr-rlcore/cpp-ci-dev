use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::usize;
use clap::Parser;
use postgres::{Client, NoTls};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct CoreIOConfig {
    coreio: CoreIOTags
}

#[derive(Debug, Deserialize)]
struct CoreIOTags {
    tags: Vec<TagConfig>,
}

#[derive(Debug, Deserialize)]
struct TagConfig {
    name: String,
    node_identifier: String,
}

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    coreio_cfg: String,

    #[arg(short, long)]
    start_time: String,

    #[arg(short, long)]
    end_time: String,

    #[arg(short, long)]
    source_table: String,

    #[arg(short, long)]
    target_table: String,
}

fn main() {
    let args = Args::parse();

    // Open the file.
    let f = File::open(args.coreio_cfg).expect("could not find coreio config");

    // Deserialize the YAML file into the Config struct.
    let config: CoreIOConfig = serde_yaml::from_reader(f).expect("serde yaml failed");
    let id_mapping: HashMap<String, String> = config.coreio.tags.into_iter().map(|tag_cfg| (tag_cfg.node_identifier, tag_cfg.name)).collect();
    let col_names: Vec<&str> = id_mapping.values().map(|c| c.as_str()).collect();

    let source_table = args.source_table;
    let target_table = args.target_table;

    let start_time = args.start_time;
    let end_time = args.end_time;

    let mut read_client = Client::connect("host=localhost user=postgres password=password", NoTls).expect("failed to create client");
    let mut write_client = Client::connect("host=localhost user=postgres password=password", NoTls).expect("failed to create client");

    let copy_in_stmt = format!("COPY (SELECT time, id, fields->>'val' as val FROM {source_table} WHERE time > '{start_time}' AND time < '{end_time}' ORDER BY time ASC) TO STDOUT");
    let instream = read_client.copy_out(&copy_in_stmt).expect("failed to open read stream");

    let col_list = col_names.join(",");
    let copy_out_stmt = format!("COPY {target_table} (time,{col_list}) FROM STDIN");
    println!("{copy_out_stmt}");
    let outstream = write_client.copy_in(&copy_out_stmt).expect("failed to open write stream");

    const MEGABYTE: usize = 1024 * 1024;
    const BUFFER_CAPACITY: usize = 16 * MEGABYTE; // 16 MB

    let buf_reader = io::BufReader::with_capacity(BUFFER_CAPACITY, instream);
    let mut buf_writer = io::BufWriter::with_capacity(BUFFER_CAPACITY, outstream);

    let mut last_timestamp = String::from("");
    let mut row: HashMap<&str, String> = Default::default(); // store data from stream here until timestamp changes

    for (i, raw_row) in buf_reader.lines().map_while(Result::ok).enumerate() {
        let mut row_elems = raw_row.split("\t");
        let timestamp = row_elems.next().expect("failed to extract timestamp from {row}");
        let id = row_elems.next().expect("failed to extract id from {row}");
        let val = row_elems.next().expect("failed to extract val from {row}");

        if i == 0 {
            last_timestamp = timestamp.to_string();
        }
        if timestamp != last_timestamp {
            // if current line has new timestamp,
            // write line data (built from previous lines) to file
            let ordered_vals: Vec<&str> = col_names.iter().map(
                |col| row.get(*col)
                .map(|v| v.as_str())
                .unwrap_or("\\N")
            ).collect();
            let new_line = last_timestamp + "\t" + &ordered_vals.join("\t") + "\n";
            buf_writer.write(&new_line.as_bytes()).unwrap();

            // update last timestamp and reset line data
            last_timestamp = timestamp.to_string();
            row.clear();
        }
        if let Some(name) = id_mapping.get(id) {
            row.insert(name.as_str(), val.to_string());
        } else {
            println!("missing id {id}\n\trow: {raw_row}\n");
        }
        // log
        if i % 1_000_000 == 0 {
            println!("processing row {i}");
        }
    }
    // write last line
    let ordered_vals: Vec<&str> = col_names.iter().map(
        |col| row.get(*col)
        .map(|v| v.as_str())
        .unwrap_or("\\N")
    ).collect();
    let new_line = last_timestamp + "\t" + &ordered_vals.join("\t") + "\n";
    buf_writer.write(&new_line.as_bytes()).unwrap();
    buf_writer.flush().unwrap();
    let outstream = buf_writer.into_inner().map_err(|e| e.into_error()).unwrap();
    let rows_written = outstream.finish().unwrap();
    println!("Successfully wrote {} rows.", rows_written);
}

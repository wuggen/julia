#![allow(unused_imports, dead_code)]

use std::fs::{self, ReadDir};
use std::io;
use std::path::{Path, PathBuf};

fn rerun_on_change<P: AsRef<Path>>(path: P) {
    println!("cargo:rerun-if-changed='{}'", path.as_ref().display());
}

fn rerun_on_any_change(dir: io::Result<ReadDir>) {
    if let Err(e) = dir {
        eprintln!("Error in directory traversal: {}", e);
    } else {
        for ent in dir.unwrap() {
            match ent {
                Ok(ent) => match ent.file_type() {
                    Ok(ty) => {
                        if ty.is_dir() {
                            rerun_on_any_change(fs::read_dir(ent.path()));
                        } else {
                            rerun_on_change(ent.path());
                        }
                    }

                    Err(e) => {
                        eprintln!("Error in directory traversal: {}", e);
                    }
                },

                Err(e) => {
                    eprintln!("Error in directory traversal: {}", e);
                }
            }
        }
    }
}

fn main() {
    //rerun_on_any_change(fs::read_dir(PathBuf::from("src")));
}

use std::fs;

use log::info;

use crate::{gui::launch, model::{Model, TrainData, Vocabulary}};
mod gui;
mod model;


fn main() {
    env_logger::init();
    launch().expect("failed to load interface");
}

use std::fs;

use log::info;

use crate::{gui::launch, model::{Model, TrainData, Vocabulary}};
mod gui;
mod model;

fn model_from_training() -> Model {
    let train_data = TrainData::from_string(fs::read_to_string("data/monte_cristo.txt").unwrap());
    info!("train data loaded");
    let mut model = Model::new(Vocabulary::from_train_data(train_data.clone()), 100);
    model.train_auto(&train_data);
    model
}

fn model_from_file() -> Model {
    Model::load_from_file("model.bin".to_string()).unwrap()
}

fn main() {
    env_logger::init();
    launch().expect("failed to load interface");
    // let model = model_from_training();

    // match model.cosine("neige".to_string()) {
    //     None => println!("word not found"),
    //     Some(list) => {
    //         list[0..10].iter().for_each(|(word, cos)| println!("{} : {}", word, cos));
    //     }
    // }

    // // model.talk();

    // model.save("model.bin".to_string()).expect("failed to save model");
}

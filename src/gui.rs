#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{env, fs};

use eframe::egui;
use egui::Label;
use crate::model::{self, Model, TrainData};



pub fn launch() -> eframe::Result {
    let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
            ..Default::default()
        };
        eframe::run_native(
            "Word2Vec trainer",
            options,
            Box::new(|cc| {
                // This gives us image support:
                Ok(Box::<MyApp>::default())
            }),
        )
}


struct MyApp {
    model_loaded: Option<Model>,
    model_path: Option<String>,
    train_data_loaded: Option<TrainData>,
    train_data_path: Option<String>,
    ephoch_count: usize,
    reference_word: String,
    words_found: Vec<String>,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            model_loaded: None,
            model_path: None,
            train_data_loaded: None,
            train_data_path: None,
            ephoch_count: 5,
            reference_word: "".to_string(),
            words_found: Vec::new(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Word2Vec trainer");
            ui.horizontal(|ui| {
                if ui.button("Open model").clicked() && let Some(path) = rfd::FileDialog::new().add_filter("model binary (.bin)", &["bin"]).set_directory(env::current_dir().unwrap()).pick_file() {
                    self.model_path = Some(path.display().to_string());
                    self.model_loaded = Some(Model::load_from_file(path.display().to_string()).expect("failed to load binary"));
                }
                if let Some(path) = self.model_path.clone() {
                    ui.label(format!("current model: {}", path));
                }
            });

            if let Some(model) = &mut self.model_loaded {
                ui.collapsing("Training", |ui| {
                    if ui.button("Open train data (plain UTF-8 text)").clicked() && let Some(path) = rfd::FileDialog::new().set_directory(env::current_dir().unwrap()).pick_file() {
                        self.train_data_path = Some(path.display().to_string());
                        self.train_data_loaded = Some(TrainData::from_string(fs::read_to_string(path.display().to_string()).expect("failed to load train data")));
                    }
                    if let Some(train_data) = &self.train_data_loaded {
                        ui.add(egui::Slider::new(&mut self.ephoch_count, 1..=120).text("epochs"));
                        if ui.button("train").clicked() {
                            model.train(train_data, self.ephoch_count);
                            println!("test")
                        }
                    }
                });

                ui.collapsing("Similar word", |ui| {

                    ui.text_edit_singleline(&mut self.reference_word);
                    if !self.reference_word.is_empty() {
                        if ui.button("Search similar").clicked() {
                            if let Some(words) = model.cosine(self.reference_word.clone()) {
                                self.words_found = Vec::new();
                                for (word, cos) in words {
                                    self.words_found.push(word + " : " + cos.to_string().as_str());
                                }
                            } else {
                                self.words_found = vec!["Word not found in model :(".to_string()];
                            }
                        }
                    }
                    for word in &self.words_found[0..10.min(self.words_found.len())] {
                        ui.label(word);
                    }
                });

            }



            if let Some(model) = self.model_loaded.clone() && ui.button("Save").clicked() && let Some(save_path) = rfd::FileDialog::new().add_filter("model binary (.bin)", &["bin"]).set_directory(env::current_dir().unwrap()).save_file() {
                model.save(save_path.display().to_string()).expect("Failed to save model");
            }

        });
    }

}

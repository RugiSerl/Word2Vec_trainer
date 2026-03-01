use std::path::is_separator;
use std::process::Stdio;
use std::{collections::HashMap, fs};
use std::io::{Read, Stdout, Write, stdout};
use log::info;
use serde::{Serialize, Deserialize};
use rand::{RngExt, rng};


#[derive(Serialize, Deserialize, Clone)]
pub struct WordVector {
    data: Vec<Vec<f32>>,
    dimension: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Vocabulary {
    ordered_data: Vec<String>,
    ids: HashMap<String, usize>,
    index: usize,
}
#[derive(Serialize, Deserialize, Clone)]
pub struct Model {
    vectors: WordVector,
    voc: Vocabulary
}

#[derive(Clone)]
pub struct TrainData(Vec<String>);


impl WordVector {
    pub fn new(word_count: usize, dimension: usize) -> WordVector {
        let mut rng = rand::rng();
        let limit = 0.5 / (dimension as f32);
        // Creating matrix of dimension wordcount * dimension
        let data = (0..word_count).into_iter().map(|_| (0..dimension).map(|_| rng.random_range(-limit..limit)).collect()).collect();
        WordVector { data, dimension }
    }
}

impl Vocabulary {
    pub fn from_train_data(text: TrainData) -> Vocabulary {
        let mut ordered_data = Vec::new();
        let mut ids = HashMap::new();
        let mut id = 0;
        text.0.iter().for_each(|word| {
            ids.entry(word.to_string()).or_insert_with(|| {
                ordered_data.push(word.to_string());
                let current_id = id;
                id += 1;
                current_id
            });
        });
        Vocabulary { ordered_data, ids, index: 0 }
    }

    pub fn len(&self) -> usize {
        self.ordered_data.len()
    }

    pub fn get_word_id(&self, word: String) -> Option<usize> {
        self.ids.get(&word).cloned()
    }
}


impl Model {
    pub fn new(vocabulary: Vocabulary, dimension: usize) -> Model {
        Model { vectors: WordVector::new(vocabulary.len(), dimension), voc: vocabulary }
    }

    pub fn load_from_file(path: String) -> bincode::Result<Model> {
        info!("Loading from file {}", path);
        let mut read_file = fs::File::open(path).expect("Unable to open file");
        let mut content = Vec::new();
        read_file.read_to_end(&mut content).expect("Reading error");
        Ok(bincode::deserialize(&content)?)
    }

    pub fn save(&self, path: String) -> bincode::Result<()> {
        info!("Saving to file {}", path);
        let mut write_file = fs::File::create(path).expect("Unable to create file");
        let encoded: Vec<u8> = bincode::serialize(&self)?;
        write_file.write_all(&encoded).expect("Writing error");
        Ok(())
    }

    fn train_step(&mut self, target_id: usize, context_id: usize, learning_rate: f32, target_label: f32) {
        // Make backup of current vectors for later use.
        let vec_target = self.vectors.data[target_id].clone();
        let vec_context = self.vectors.data[context_id].clone();

        let dot: f32 = vec_target.iter().zip(vec_context.iter()).map(|(a, b)| a*b).sum();

        let prediction = 1.0 / (1.0 + f32::exp(-dot));

        let error = target_label - prediction;

        for i in 0..self.vectors.dimension {
            self.vectors.data[target_id][i] += learning_rate * error * vec_context[i];
            self.vectors.data[context_id][i] += learning_rate * error * vec_target[i];
        }
    }

    pub fn train_epoch(&mut self, data: &TrainData, learning_rate: f32) {
        let window_size = 2;
        let train_data_id = data.to_id_vec(&self.voc);
        let mut rng = rng();

        for (i, &target_id) in train_data_id.iter().enumerate() {
            info!("training word {} / {} (learning rate : {})", i, train_data_id.len(), learning_rate);
            let window_start = if i < window_size {0} else {i-window_size};
            let window_end = if i+window_size >= train_data_id.len()-1 {train_data_id.len()-1} else {i+window_size};
            for j in window_start..=window_end {
                if i != j {
                    let context_id = train_data_id[j];

                    // Positive sampling.
                    self.train_step(target_id, context_id, learning_rate, 1.0);

                    // Negative sampling.
                    for _ in 0..5 {
                        let random_id = rng.random_range(0..self.voc.len());
                        if random_id != target_id {
                            self.train_step(target_id, random_id, learning_rate, 0.0);
                        }
                    }

                }
            }
        }
    }

    pub fn train(&mut self, data: &TrainData, epochs: usize) {
        let mut learning_rate = 0.05;
        let decay = learning_rate / epochs as f32;

        for epoch in 1..=epochs {
            self.train_epoch(data, learning_rate);
            learning_rate -= decay;
        }
    }

    pub fn train_auto(&mut self, data: &TrainData) {
        self.train(data, Vocabulary::from_train_data(data.clone()).len().ilog10() as usize * 4);
    }

    pub fn cosine(&self, word: String) -> Option<Vec<(String, f32)>> {
        let word_id = self.voc.get_word_id(word)?;
        let word_vec = &self.vectors.data[word_id];
        let norm_word = word_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut scores: Vec<(usize, f32)> = self.vectors.data.iter().enumerate().map(|(i, vec)| {
            let dot: f32 = vec.iter().zip(word_vec).map(|(a, b)| a * b).sum();
            let norm_v = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let similarity = dot / (norm_v * norm_word + 1e-8); // +1e-8 pour éviter division par zéro
            (i, similarity)
        }).collect();

        scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        Some(scores.into_iter()
            .map(|(i, val)| (self.voc.ordered_data[i].clone(), val))
            .collect())
    }

    pub fn talk(&self) {
        let mut rng = rand::rng();
        let word = self.voc.ordered_data[rng.random_range(0..self.voc.ordered_data.len())].clone();
        let mut out = stdout();

        loop {
            let word = self.cosine(word.clone()).unwrap().get(rng.random_range(0..5)).unwrap().to_owned().0;
            print!("{} ", word);
            out.flush();
        }
    }

}

impl TrainData {
    pub fn from_string(data: String) -> TrainData {
        let data = data.split(TrainData::is_separator).map(|s| s.chars().filter(TrainData::is_relevant).map(|c|c.to_lowercase().to_string()).collect()).collect();
        TrainData(data)
    }

    pub fn to_id_vec(&self, voc: &Vocabulary) -> Vec<usize> {
        self.0.iter().filter_map(|x| voc.get_word_id(x.to_owned())).collect()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_separator(c: char) -> bool {
        c==' '|| c=='.' || c==',' || c=='\'' || c=='\n' || c== '-'
    }

    pub fn is_relevant(c: &char) -> bool {
        c.is_alphabetic()
    }

}




impl Iterator for Vocabulary {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.ordered_data.len() {
            return None
        } else {
            self.index += 1;
            return Some(self.ordered_data[self.index - 1].clone())
        }
    }
}

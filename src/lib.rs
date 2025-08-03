use crate::math::{random_normal_matrix, sigmoid};

mod math;

type ActivationFunction = fn(f64) -> f64;

struct NeuralNetwork {
    weights_in_hid: Vec<Vec<f64>>,
    weights_hid_out: Vec<Vec<f64>>,
    learning_rate: f64,
    activation_fn: ActivationFunction
}

impl NeuralNetwork {
    pub fn new(
        input_nodes_number: usize,
        hidden_nodes_number: usize,
        output_nodes_number: usize,
        learning_rate : f64
    ) -> NeuralNetwork {
        let weights_in_hid = random_normal_matrix(
            0.0, 
            (hidden_nodes_number as f64).powf(-0.5), 
            (hidden_nodes_number, input_nodes_number)
        );
        let weights_hid_out = random_normal_matrix(
            0.0, 
            (output_nodes_number as f64).powf(-0.5), 
            (output_nodes_number, hidden_nodes_number)
        );
        NeuralNetwork { weights_in_hid, weights_hid_out, learning_rate, activation_fn: sigmoid }
    }
}
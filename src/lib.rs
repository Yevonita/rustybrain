use crate::math::{dot_product, matrix_add, matrix_elementwise_mul, matrix_scale, matrix_subtract, random_normal_matrix, sigmoid, sigmoid_derivative, traspose_matrix};

mod math;

type ActivationFunction = fn(Vec<Vec<f64>>) -> Vec<Vec<f64>>;

pub struct NeuralNetwork {
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
            0.001, // Much smaller standard deviation
            (hidden_nodes_number, input_nodes_number)
        );
        let weights_hid_out = random_normal_matrix(
            0.0, 
            0.001, // Much smaller standard deviation
            (output_nodes_number, hidden_nodes_number)
        );
        let activation_fn = |x: Vec<Vec<f64>>| -> Vec<Vec<f64>> {
            x.into_iter()
                .map(|row| row.into_iter().map(sigmoid).collect())
                .collect()
        };
        NeuralNetwork { weights_in_hid, weights_hid_out, learning_rate, activation_fn}
    }

    pub fn query(&self, input_list: Vec<f64>) -> Vec<Vec<f64>> {
        let inputs = traspose_matrix(&vec![input_list]);
        let hidden_inputs = dot_product(&self.weights_in_hid, &inputs);
        let hidden_outputs = (self.activation_fn)(hidden_inputs);
        let final_inputs = dot_product(&self.weights_hid_out, &hidden_outputs);
        let final_outputs = (self.activation_fn)(final_inputs);
        final_outputs
    }

    pub fn train(&mut self, input_list: Vec<f64>, target_list: Vec<f64>) {
        let inputs = traspose_matrix(&vec![input_list]);
        let targets = traspose_matrix(&vec![target_list]);

        // Forward pass
        let hidden_inputs = dot_product(&self.weights_in_hid, &inputs);
        let hidden_outputs = (self.activation_fn)(hidden_inputs);
        let final_inputs = dot_product(&self.weights_hid_out, &hidden_outputs);
        let final_outputs = (self.activation_fn)(final_inputs);

        // Output error
        let output_errors = matrix_subtract(&targets, &final_outputs);

        // Output gradient
        let output_grad = matrix_elementwise_mul(
            &output_errors,
            &sigmoid_derivative(&final_outputs)
        );
        
        // Update weights from hidden to output
        let delta_weights_hid_out = matrix_scale(
            &dot_product(&output_grad, &traspose_matrix(&hidden_outputs)),
            self.learning_rate
        );
        self.weights_hid_out = matrix_add(&self.weights_hid_out, &delta_weights_hid_out);

        // Hidden layer error - fixed calculation
        let hidden_errors = dot_product(&traspose_matrix(&self.weights_hid_out), &output_grad);

        // Hidden gradient
        let hidden_grad = matrix_elementwise_mul(
            &hidden_errors,
            &sigmoid_derivative(&hidden_outputs)
        );
        
        // Update weights from input to hidden
        let delta_weights_in_hid = matrix_scale(
            &dot_product(&hidden_grad, &traspose_matrix(&inputs)),
            self.learning_rate
        );
        self.weights_in_hid = matrix_add(&self.weights_in_hid, &delta_weights_in_hid);
    }
}
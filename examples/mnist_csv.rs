use rustybrain::NeuralNetwork;
use csv::ReaderBuilder;
use std::error::Error;

fn read_csv(path: &str) -> Result<Vec<(Vec<f64>, Vec<f64>)>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(path)?;
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result?;
        // Assume first column is label, rest are features
        let label: usize = record[0].parse()?;
        let input: Vec<f64> = record.iter().skip(1).map(|x| x.parse::<f64>().unwrap() / 255.0).collect();
        // One-hot encode the label
        let mut target = vec![0.0; 10];
        target[label] = 1.0;
        data.push((input, target));
    }
    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Create neural network: 784 inputs (28x28 pixels), 200 hidden nodes, 10 outputs (digits 0-9)
    let mut nn = NeuralNetwork::new(784, 200, 10, 0.3);

    // Read training data
    let train_data = read_csv("examples/data/mnist_train_100.csv")?;
    println!("Training on {} samples", train_data.len());

    // Train for multiple epochs
    for epoch in 1..=50 {
        if epoch % 10 == 0 {
            println!("Epoch {}", epoch);
        }
        for (input, target) in train_data.iter() {
            nn.train(input.clone(), target.clone());
        }
    }

    // Read test data
    let test_data = read_csv("examples/data/mnist_test_10.csv")?;
    println!("Testing on {} samples", test_data.len());

    // Test
    let mut correct = 0;
    for (input, target) in &test_data {
        let output_matrix = nn.query(input.clone());
        
        // Extract all output values - the matrix is 10x1, so we need to get all 10 values
        let output: Vec<f64> = output_matrix.iter().map(|row| row[0]).collect();
        
        let guess = output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let actual = target.iter().position(|&x| x == 1.0).unwrap();
        if guess == actual {
            correct += 1;
        }
        println!("Predicted: {}, Actual: {}", guess, actual);
    }
    
    println!("Accuracy: {:.2}%", (correct as f64 / test_data.len() as f64) * 100.0);
    Ok(())
}
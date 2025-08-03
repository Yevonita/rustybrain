use rand_distr::{Normal, Distribution};

pub fn random_normal_matrix(mean: f64, dev: f64, size: (usize, usize)) -> Vec<Vec<f64>> {
    let normal = Normal::new(mean, dev).unwrap();
    let mut rng = rand::rng();
    let mut matrix = Vec::with_capacity(size.0);
    for _ in 0..size.0 {
        let mut row = Vec::with_capacity(size.1);
        for _ in 0..size.1 {
            row.push(normal.sample(&mut rng));
        }
        matrix.push(row);
    }
    matrix
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let input = 0.4;
        let expected = 0.598687660112452;
        let actual = sigmoid(input);
        assert_eq!(expected, actual, "Expected result from sigmoid of 0.4 is 0.598687660112452");
    }
}
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

pub fn traspose_matrix(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return vec![];
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

pub fn dot_product(matrix_a: &Vec<Vec<f64>>, matrix_b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows_a = matrix_a.len();
    if rows_a == 0 {
        return vec![];  
    }
    let cols_a = matrix_a[0].len();
    let rows_b = matrix_b.len();
    if rows_b == 0 {
        return vec![];
    }
    let cols_b = matrix_b[0].len();
    if cols_a != rows_b {
        panic!("Incompatible dimensions {}x{} vs {}x{}", rows_a, cols_a, rows_b, cols_b);
    }

    let mut result = vec![vec![0.0; cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = 0.0;
            for k in 0..cols_a {
                sum += matrix_a[i][k] * matrix_b[k][j];
            }
            result[i][j] = sum;
        }
    }
    result

}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_sigmoid() {
        let input = 0.4;
        let expected = 0.598687660112452;
        let actual = sigmoid(input);
        assert_eq!(expected, actual, "Expected result from sigmoid of 0.4 is 0.598687660112452");
    }

    #[test]
    fn test_transpose_matrix() {
        let input = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0]
        ];
        let expected = vec![
            vec![1.0, 4.0],
            vec![2.0, 5.0],
            vec![3.0, 6.0],
        ];
        let actual = traspose_matrix(&input);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![vec![1.14062627, 0.10998959, 0.1515219]];
        let b = vec![vec![1.0], vec![2.0], vec![3.0]];
        let expected = vec![vec![1.8151711499999998]];
        let actual = dot_product(&a, &b);
        assert_eq!(expected, actual);
    }
}
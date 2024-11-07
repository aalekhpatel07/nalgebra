use rand_package::Rng;
use nalgebra::Const;
use nalgebra_glm::{RealNumber, TMat};

/// Generate a random matrix of all zeroes
/// except one (+1/-1) at some row in every column.
fn random_clarkson_woodruff_matrix<
    T: RealNumber,
    const ROWS: usize,
    const COLS: usize,
    R: Rng + ?Sized,
>(rng: &mut R) -> TMat<T, ROWS, COLS> {

    let mut zeroes = TMat::from_element_generic(Const::<ROWS>, Const::<COLS>, T::zero());

    (0..COLS)
        .into_iter()
        .for_each(|column_index| {
            let row_index = rng.gen_range(0..ROWS);
            let negative: bool = rng.gen();
            let mut value = T::one();
            if negative {
                value = value.neg();
            }
            unsafe {
                *zeroes.get_unchecked_mut((row_index, column_index)) = value;
            }
        });

    zeroes
}

/// Applies a Clarkson-Woodruff Transform/sketch to the input matrix.
///
/// Given an input matrix ``A`` of size ``(ROWS, COLS)``, compute a matrix ``A'``
/// of size `(SKETCH, COLS)` (i.e. a reduction in row dimensionality) such that
///
/// $ |Ax| \approx |A'x| $
///
/// with high probability via the Clarkson-Woodruff Transform, otherwise known
/// as the CountSketch matrix.
pub fn clarkson_woodruff_transform<
    T: RealNumber,
    const ROWS: usize,
    const COLS: usize,
    const SKETCH: usize,
    R: Rng + ?Sized,
> (
    matrix: &TMat<T, ROWS, COLS>,
    rng: &mut R
) -> TMat<T, SKETCH, COLS> {
    random_clarkson_woodruff_matrix::<T, SKETCH, ROWS, R>(rng) * matrix
}


#[cfg(test)]
mod tests {
    use rand_package::distributions::Standard;
    use rand_package::Rng;
    use nalgebra_glm::TMat;
    use super::clarkson_woodruff_transform;

    #[test]
    fn test_clarkson_woodruff_matrix() {
        let mut rng = rand_package::thread_rng();

        const ROWS: usize = 10;
        const COLS: usize = 10;
        const SKETCH: usize = 5;

        let mut random_values: [[f64; ROWS]; COLS] = [[0.; ROWS]; COLS];

        for row in 0..ROWS {
            for col in 0..COLS {
                random_values[row][col] = rng.gen();
            }
        }

        // Choose some matrix A and vector B.
        let matrix = TMat::<f64, ROWS, COLS>::from_distribution(&Standard, &mut rng);
        let sketched = clarkson_woodruff_transform::<_, ROWS, COLS, SKETCH, _>(&matrix, &mut rng);

        let samples = 1_000_000;
        let mut differences = Vec::with_capacity(samples);
        for _ in 0..samples {
            let vector = TMat::<f64, COLS, 1>::from_distribution(&Standard, &mut rng);

            let og_product = matrix * vector;
            let sketched_product = sketched * vector;

            let difference = (og_product.norm() - sketched_product.norm()).abs();
            differences.push(difference);
        }

        differences.sort_unstable_by(f64::total_cmp);

        let sum: f64 = differences.iter().sum();
        let mean: f64 = sum / differences.len() as f64;
        let median: f64 = (differences[samples / 2] + differences[samples / 2 + 1]) / 2.0;
        let largest: f64 = differences[samples - 1];
        let smallest: f64 = differences[0];
        let variance: f64 = differences.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (samples as f64 - 1.0);
        let stddev: f64 = variance.sqrt();
        let iqr: f64 = differences[(3 * samples) / 4] - differences[samples / 4];
        let q1: f64 = median - 1.5 * iqr;
        let q3: f64 = median + 1.5 * iqr;
        let outliers = differences.iter().filter(|&&v| v < q1 || v > q3).count();
        let outlier_ratio = ((outliers as f64) / samples as f64) * 100.0;

        eprintln!(
            "mean: {mean:.8}, median: {median:.8}, \
            smallest: {smallest:.8}, largest: {largest:.8}, \
            variance: {variance:.8}, stddev: {stddev:.8} \
            iqr: {iqr:.8}, \
            q1: {q1:.8}, \
            q3: {q3:.8}, \
            outliers: {outliers}, \
            outliers_percent: {outlier_ratio:.8},
            "
        );

        assert!(
            outlier_ratio <= 5.,
            "not expecting too many outliers since this is supposed to be a good approximation that preseves magnitudes."
        );
        assert!(
            (mean - median).abs() <= 1e-1,
            "mean and median should be pretty close."
        );
    }
}

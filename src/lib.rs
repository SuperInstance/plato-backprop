use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single prompt correction event.
/// In the PLATO signal chain, L4 (cloud) corrections become few-shot examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCorrection {
    pub tile_id: Uuid,
    pub predicted: String,
    pub actual: String,
    pub error: f64,
    pub timestamp: u64,
}

impl PromptCorrection {
    /// Compute string distance error between predicted and actual.
    /// Uses normalized Levenshtein distance: 0.0 = identical, 1.0 = maximally different.
    pub fn error(predicted: &str, actual: &str) -> f64 {
        if predicted.is_empty() && actual.is_empty() {
            return 0.0;
        }
        if predicted.is_empty() || actual.is_empty() {
            return 1.0;
        }

        let p: Vec<char> = predicted.chars().collect();
        let a: Vec<char> = actual.chars().collect();
        let plen = p.len();
        let alen = a.len();

        let mut dp = vec![vec![0usize; alen + 1]; plen + 1];
        for i in 0..=plen {
            dp[i][0] = i;
        }
        for j in 0..=alen {
            dp[0][j] = j;
        }
        for i in 1..=plen {
            for j in 1..=alen {
                if p[i - 1] == a[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
                }
            }
        }

        let max_len = plen.max(alen) as f64;
        dp[plen][alen] as f64 / max_len
    }

    pub fn new(tile_id: Uuid, predicted: &str, actual: &str, timestamp: u64) -> Self {
        Self {
            tile_id,
            predicted: predicted.to_string(),
            actual: actual.to_string(),
            error: Self::error(predicted, actual),
            timestamp,
        }
    }
}

/// A sliding window of prompt corrections — the "prompt batch" for gradient descent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptWindow {
    pub corrections: Vec<PromptCorrection>,
    pub max_size: usize,
    /// Decay rate lambda (default 0.1). Controls exponential decay of influence.
    pub decay_rate: f64,
}

impl PromptWindow {
    pub fn new(max_size: usize) -> Self {
        Self {
            corrections: Vec::new(),
            max_size,
            decay_rate: 0.1,
        }
    }

    /// Add a correction. Evicts oldest if window is full.
    pub fn push(&mut self, correction: PromptCorrection) {
        if self.corrections.len() >= self.max_size {
            self.corrections.remove(0);
        }
        self.corrections.push(correction);
    }

    /// Reduce influence of old corrections by applying decay based on position.
    /// This simulates the aging of few-shot examples in the prompt window.
    pub fn decay(&mut self, factor: f64) {
        // Apply a multiplicative decay to error values of existing corrections
        for correction in &mut self.corrections {
            correction.error *= factor;
        }
    }

    /// Compute gradient traces: each correction's influence on the current prediction.
    /// Influence = e^(-lambda * age), where age = distance from end of window.
    pub fn compute_gradient(&self) -> Vec<GradientTrace> {
        let n = self.corrections.len();
        if n == 0 {
            return Vec::new();
        }

        self.corrections
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let age = (n - 1 - i) as f64;
                let influence = (-self.decay_rate * age).exp();
                GradientTrace {
                    correction_id: c.tile_id,
                    influence_on_prediction: influence * c.error,
                    decay_factor: influence,
                }
            })
            .collect()
    }

    /// Sum of all influences — the total gradient magnitude.
    pub fn total_gradient(&self) -> f64 {
        self.compute_gradient()
            .iter()
            .map(|g| g.influence_on_prediction)
            .sum()
    }

    /// Effective learning rate = total_gradient / window_size.
    /// Higher means the prompt is "teaching" more aggressively.
    pub fn effective_learning_rate(&self) -> f64 {
        if self.corrections.is_empty() {
            return 0.0;
        }
        self.total_gradient() / self.corrections.len() as f64
    }

    /// Check if the prompt has converged (gradient near zero).
    pub fn is_converged(&self, threshold: f64) -> bool {
        self.total_gradient() < threshold
    }
}

/// A single correction's influence trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientTrace {
    pub correction_id: Uuid,
    pub influence_on_prediction: f64,
    pub decay_factor: f64,
}

/// Full backpropagation result for a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpropResult {
    pub prediction_id: Uuid,
    pub influenced_by: Vec<GradientTrace>,
    pub total_gradient: f64,
}

impl BackpropResult {
    pub fn from_window(window: &PromptWindow, prediction_id: Uuid) -> Self {
        let traces = window.compute_gradient();
        let total = traces.iter().map(|g| g.influence_on_prediction).sum();
        Self {
            prediction_id,
            influenced_by: traces,
            total_gradient: total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_correction(predicted: &str, actual: &str) -> PromptCorrection {
        PromptCorrection::new(Uuid::new_v4(), predicted, actual, 0)
    }

    #[test]
    fn test_push_basic() {
        let mut window = PromptWindow::new(5);
        window.push(make_correction("a", "b"));
        assert_eq!(window.corrections.len(), 1);
    }

    #[test]
    fn test_eviction_when_full() {
        let mut window = PromptWindow::new(3);
        let ids: Vec<Uuid> = (0..5)
            .map(|_| {
                let c = make_correction("a", "b");
                let id = c.tile_id;
                window.push(c);
                id
            })
            .collect();
        assert_eq!(window.corrections.len(), 3);
        // First two should be evicted
        assert_ne!(window.corrections[0].tile_id, ids[0]);
        assert_ne!(window.corrections[0].tile_id, ids[1]);
        assert_eq!(window.corrections[0].tile_id, ids[2]);
    }

    #[test]
    fn test_window_overflow() {
        let mut window = PromptWindow::new(2);
        window.push(make_correction("a", "b"));
        window.push(make_correction("c", "d"));
        window.push(make_correction("e", "f"));
        assert_eq!(window.corrections.len(), 2);
        assert_eq!(window.corrections[0].predicted, "c");
        assert_eq!(window.corrections[1].predicted, "e");
    }

    #[test]
    fn test_decay_reduces_influence() {
        let mut window = PromptWindow::new(5);
        window.push(make_correction("hello", "world"));
        let before = window.total_gradient();
        window.decay(0.5);
        let after = window.total_gradient();
        assert!(after < before);
    }

    #[test]
    fn test_gradient_computation_correctness() {
        let mut window = PromptWindow::new(5);
        window.decay_rate = 0.0; // No decay — all influences should be 1.0
        window.push(PromptCorrection::new(Uuid::new_v4(), "a", "b", 0));
        // error("a","b") = 1.0 (1 edit / max(1,1))
        let traces = window.compute_gradient();
        assert_eq!(traces.len(), 1);
        assert!((traces[0].decay_factor - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_gradient_decreases_with_age() {
        let mut window = PromptWindow::new(5);
        window.decay_rate = 0.1;
        // Push identical-error corrections
        for _ in 0..4 {
            window.push(PromptCorrection::new(Uuid::new_v4(), "a", "b", 0));
        }
        let traces = window.compute_gradient();
        // Newest (last) should have highest influence
        let last = traces.last().unwrap().influence_on_prediction;
        let first = traces.first().unwrap().influence_on_prediction;
        assert!(last > first);
    }

    #[test]
    fn test_total_gradient_decreases_with_convergence() {
        let mut window = PromptWindow::new(5);
        window.push(make_correction("cat", "dog"));
        let g1 = window.total_gradient();
        window.push(make_correction("cat", "cat")); // error = 0
        let g2 = window.total_gradient();
        assert!(g2 > 0.0); // Still has the old correction
        // Push more zero-error corrections to dilute
        for _ in 0..10 {
            window.push(make_correction("cat", "cat"));
        }
        let g3 = window.total_gradient();
        assert!(g3 < g1);
    }

    #[test]
    fn test_effective_learning_rate() {
        let mut window = PromptWindow::new(5);
        assert_eq!(window.effective_learning_rate(), 0.0); // empty

        window.push(make_correction("a", "b"));
        let lr = window.effective_learning_rate();
        assert!(lr > 0.0);

        // lr = total_gradient / window_size
        let expected = window.total_gradient() / 1.0;
        assert!((lr - expected).abs() < 1e-9);
    }

    #[test]
    fn test_convergence_detection() {
        let mut window = PromptWindow::new(5);
        // Empty window is converged
        assert!(window.is_converged(0.01));

        // Add near-identical corrections
        window.push(make_correction("hello", "hello"));
        assert!(window.is_converged(0.01)); // error = 0
    }

    #[test]
    fn test_error_identical_is_zero() {
        let e = PromptCorrection::error("hello", "hello");
        assert_eq!(e, 0.0);
    }

    #[test]
    fn test_error_different_is_positive() {
        let e = PromptCorrection::error("abc", "xyz");
        assert!(e > 0.0);
    }

    #[test]
    fn test_error_empty_strings() {
        assert_eq!(PromptCorrection::error("", ""), 0.0);
        assert_eq!(PromptCorrection::error("a", ""), 1.0);
        assert_eq!(PromptCorrection::error("", "a"), 1.0);
    }

    #[test]
    fn test_backprop_result_from_window() {
        let mut window = PromptWindow::new(5);
        window.push(make_correction("a", "b"));
        window.push(make_correction("c", "d"));
        let pred_id = Uuid::new_v4();
        let result = BackpropResult::from_window(&window, pred_id);
        assert_eq!(result.prediction_id, pred_id);
        assert_eq!(result.influenced_by.len(), 2);
        assert!((result.total_gradient - window.total_gradient()).abs() < 1e-9);
    }

    #[test]
    fn test_empty_window() {
        let window = PromptWindow::new(5);
        assert!(window.compute_gradient().is_empty());
        assert_eq!(window.total_gradient(), 0.0);
        assert_eq!(window.effective_learning_rate(), 0.0);
        assert!(window.is_converged(0.001));
    }

    #[test]
    fn test_single_correction() {
        let mut window = PromptWindow::new(5);
        window.push(make_correction("abc", "abd"));
        let traces = window.compute_gradient();
        assert_eq!(traces.len(), 1);
        assert!((traces[0].decay_factor - 1.0).abs() < 1e-9); // newest = full influence
    }

    #[test]
    fn test_all_identical_corrections() {
        let mut window = PromptWindow::new(5);
        for _ in 0..5 {
            window.push(make_correction("hello", "hello"));
        }
        // All errors = 0, so all influences should be 0
        let traces = window.compute_gradient();
        for t in &traces {
            assert!(t.influence_on_prediction.abs() < 1e-9);
        }
        assert_eq!(window.total_gradient(), 0.0);
    }

    #[test]
    fn test_decay_rate_sensitivity() {
        let mut fast = PromptWindow::new(10);
        let mut slow = PromptWindow::new(10);
        fast.decay_rate = 1.0;
        slow.decay_rate = 0.01;
        for _ in 0..5 {
            let c = make_correction("a", "b");
            fast.push(c.clone());
            slow.push(c);
        }
        // Fast decay should have lower total gradient
        assert!(fast.total_gradient() < slow.total_gradient());
    }

    #[test]
    fn test_fast_vs_slow_decay_convergence() {
        let mut fast = PromptWindow::new(10);
        let mut slow = PromptWindow::new(10);
        fast.decay_rate = 1.0;
        slow.decay_rate = 0.01;

        // Add a correction, then many correct ones
        fast.push(make_correction("a", "b"));
        slow.push(make_correction("a", "b"));
        for _ in 0..9 {
            fast.push(make_correction("a", "a"));
            slow.push(make_correction("a", "a"));
        }

        // Fast decay converges quicker (old error decays fast)
        assert!(fast.is_converged(0.01));
        // Slow decay still influenced by old error
        assert!(fast.total_gradient() < slow.total_gradient());
    }

    #[test]
    fn test_decay_method_reduces_error_values() {
        let mut window = PromptWindow::new(5);
        window.push(make_correction("a", "b"));
        let e_before = window.corrections[0].error;
        window.decay(0.5);
        let e_after = window.corrections[0].error;
        assert!((e_after - e_before * 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_gradient_with_nonzero_decay() {
        let mut window = PromptWindow::new(5);
        window.decay_rate = 0.1;
        for i in 0..3 {
            window.push(PromptCorrection::new(Uuid::new_v4(), "a", "b", i));
        }
        let traces = window.compute_gradient();
        // Verify exponential decay ordering: newest > middle > oldest
        assert!(traces[2].decay_factor > traces[1].decay_factor);
        assert!(traces[1].decay_factor > traces[0].decay_factor);
    }
}

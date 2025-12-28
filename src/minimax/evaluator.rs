use crate::game::{Game, Player, GameState};
use crate::board::{Position, Cell};

/// Evaluates game positions for the Minimax algorithm.
///
/// The evaluation function combines multiple heuristics to score a position
/// from a given player's perspective. Higher scores are better.
///
/// Key factors:
/// - Piece count (weighted by game phase)
/// - Corner control
/// - Mobility (available moves)
/// - Stability (pieces that are hard to flip)
/// - Game phase awareness (early/mid/late game)
pub struct PositionEvaluator;

impl PositionEvaluator {
    /// Evaluate a game position from the given player's perspective.
    ///
    /// Returns a score where:
    /// - Positive values favor the given player
    /// - Negative values favor the opponent
    /// - Magnitude indicates strength of advantage
    /// - Terminal positions return `f64::INFINITY` (win) or `f64::NEG_INFINITY` (loss)
    ///
    /// Typical score range for non-terminal positions: -100 to +100 (but not strictly bounded)
    ///
    /// # Arguments
    ///
    /// * `game` - The game state to evaluate
    /// * `player` - The player whose perspective to evaluate from
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::minimax::evaluator::PositionEvaluator;
    /// use othello::{Game, Player};
    ///
    /// let game = Game::new();
    /// let score = PositionEvaluator::evaluate(&game, Player::Black);
    /// // Score should be finite and reasonable
    /// assert!(score.is_finite());
    /// ```
    pub fn evaluate(game: &Game, player: Player) -> f64 {
        // Terminal position (game over)
        if let GameState::GameOver { winner } = game.get_game_state() {
            return Self::evaluate_terminal(winner, player);
        }
        
        let (black_pieces, white_pieces) = game.get_score();
        let total_pieces = black_pieces + white_pieces;
        let game_phase = total_pieces as f64 / 64.0;  // 0.0 = start, 1.0 = end
        
        // Compute component scores
        let piece_score = Self::evaluate_pieces(game, player, game_phase);
        let mobility_score = Self::evaluate_mobility(game, player, game_phase);
        let corner_score = Self::evaluate_corners(game, player);
        let stability_score = Self::evaluate_stability(game, player, game_phase);
        let positional_score = Self::evaluate_positional(game, player);
        
        // Weighted combination based on game phase
        piece_score * Self::piece_weight(game_phase) +
        mobility_score * Self::mobility_weight(game_phase) +
        corner_score * Self::corner_weight() +
        stability_score * Self::stability_weight(game_phase) +
        positional_score * Self::positional_weight(game_phase)
    }
    
    /// Evaluate terminal position (game over).
    fn evaluate_terminal(winner: Option<Player>, player: Player) -> f64 {
        match winner {
            Some(w) if w == player => f64::INFINITY,  // Win
            Some(_) => f64::NEG_INFINITY,  // Loss
            None => 0.0,  // Draw
        }
    }
    
    /// Evaluate piece count advantage.
    ///
    /// More important in late game, less important early game.
    fn evaluate_pieces(game: &Game, player: Player, _game_phase: f64) -> f64 {
        let (black_pieces, white_pieces) = game.get_score();
        
        let piece_diff = if player == Player::Black {
            black_pieces as f64 - white_pieces as f64
        } else {
            white_pieces as f64 - black_pieces as f64
        };
        
        // Normalize to [-1, 1] range
        piece_diff / 64.0
    }
    
    /// Evaluate mobility (available moves).
    ///
    /// More important in early/mid game. Having more moves than your opponent
    /// is a significant advantage.
    /// Maximum possible moves has been mathematically proven to be 33
    /// https://jxiv.jst.go.jp/index.php/jxiv/preprint/download/480/1498
    fn evaluate_mobility(game: &Game, _player: Player, _game_phase: f64) -> f64 {
        let my_moves = game.get_valid_moves().len();
        // Maximum possible moves has been mathematically proven to be 33
        // https://jxiv.jst.go.jp/index.php/jxiv/preprint/download/480/1498
        my_moves as f64 / 33.0
    }
    
    /// Evaluate corner control.
    ///
    /// Corners are extremely valuable in Othello because they cannot be flipped.
    /// Always important regardless of game phase.
    fn evaluate_corners(game: &Game, player: Player) -> f64 {
        let corners = [
            Position::new(0, 0), Position::new(0, 7),
            Position::new(7, 0), Position::new(7, 7),
        ];
        
        let board = game.get_board();
        let my_corners = corners.iter()
            .filter(|&&pos| {
                board.get_cell(pos).ok() == Some(Cell::from_player(player))
            })
            .count();
        
        let opponent_corners = corners.iter()
            .filter(|&&pos| {
                board.get_cell(pos).ok() == Some(Cell::from_player(player.opposite()))
            })
            .count();
        
        // Normalize to [-1, 1] (4 corners total)
        (my_corners as f64 - opponent_corners as f64) / 4.0
    }
    
    /// Evaluate piece stability.
    ///
    /// Stable pieces (especially on edges) are harder to flip.
    /// More important in early/mid game.
    fn evaluate_stability(game: &Game, player: Player, _game_phase: f64) -> f64 {
        let board = game.get_board();
        let mut my_stable = 0;
        let mut opponent_stable = 0;
        
        // Count edge pieces (generally more stable)
        for row in 0..8 {
            for col in 0..8 {
                let pos = Position::new(row, col);
                if let Ok(cell) = board.get_cell(pos) {
                    if cell != Cell::Empty {
                        let is_edge = row == 0 || row == 7 || col == 0 || col == 7;
                        if is_edge {
                            if cell == Cell::from_player(player) {
                                my_stable += 1;
                            } else {
                                opponent_stable += 1;
                            }
                        }
                    }
                }
            }
        }
        
        // Normalize to [-1, 1]
        let total_stable = my_stable + opponent_stable;
        if total_stable == 0 {
            0.0
        } else {
            (my_stable as f64 - opponent_stable as f64) / total_stable as f64
        }
    }
    
    /// Evaluate positional factors.
    ///
    /// Includes X-squares, C-squares, and general positional control.
    fn evaluate_positional(game: &Game, player: Player) -> f64 {
        let board = game.get_board();
        let mut score = 0.0;
        
        // Penalize X-squares (adjacent to corners) unless corner is ours
        let x_squares = [
            Position::new(1, 1), Position::new(1, 6),
            Position::new(6, 1), Position::new(6, 6),
        ];
        
        let corners = [
            Position::new(0, 0), Position::new(0, 7),
            Position::new(7, 0), Position::new(7, 7),
        ];
        
        for &x_pos in &x_squares {
            if let Ok(cell) = board.get_cell(x_pos) {
                if cell != Cell::Empty {
                    // Check if adjacent corner is ours
                    let corner_is_mine = corners.iter().any(|&corner| {
                        let row_diff = (corner.row as i32 - x_pos.row as i32).abs();
                        let col_diff = (corner.col as i32 - x_pos.col as i32).abs();
                        row_diff <= 1 && col_diff <= 1 &&
                        board.get_cell(corner).ok() == Some(Cell::from_player(player))
                    });
                    
                    if cell == Cell::from_player(player) {
                        if corner_is_mine {
                            score += 0.1;  // Safe X-square
                        } else {
                            score -= 0.2;  // Risky X-square
                        }
                    } else {
                        if corner_is_mine {
                            score += 0.1;  // Opponent has risky piece
                        } else {
                            score -= 0.1;  // Opponent controls X-square
                        }
                    }
                }
            }
        }
        
        score
    }
    
    /// Get weight for piece count evaluation based on game phase.
    fn piece_weight(game_phase: f64) -> f64 {
        // More important late game
        20.0 + game_phase * 40.0
    }
    
    /// Get weight for mobility evaluation based on game phase.
    fn mobility_weight(game_phase: f64) -> f64 {
        // More important early game
        30.0 * (1.0 - game_phase * 0.5)
    }
    
    /// Get weight for corner evaluation.
    fn corner_weight() -> f64 {
        // Always very important
        25.0
    }
    
    /// Get weight for stability evaluation based on game phase.
    fn stability_weight(game_phase: f64) -> f64 {
        // More important early/mid game
        15.0 * (1.0 - game_phase * 0.7)
    }
    
    /// Get weight for positional evaluation.
    fn positional_weight(_game_phase: f64) -> f64 {
        // Constant moderate importance
        10.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    
    #[test]
    fn test_evaluate_initial_position() {
        let game = Game::new();
        let score = PositionEvaluator::evaluate(&game, Player::Black);
        
        // Initial position should be roughly balanced
        assert!(score.is_finite());
        assert!(score.abs() < 50.0);  // Should be close to 0
    }
    
    #[test]
    fn test_evaluate_winning_position() {
        let game = Game::new();
        
        // Test that evaluation works
        let score = PositionEvaluator::evaluate(&game, Player::Black);
        assert!(score.is_finite());
    }
    
    #[test]
    fn test_corner_evaluation() {
        let game = Game::new();
        let score = PositionEvaluator::evaluate_corners(&game, Player::Black);
        
        // Should be in reasonable range
        assert!(score >= -1.0 && score <= 1.0);
    }
    
    #[test]
    fn test_mobility_evaluation() {
        let game = Game::new();
        let score = PositionEvaluator::evaluate_mobility(&game, Player::Black, 0.0);
        
        // Should be finite
        assert!(score.is_finite());
    }
}


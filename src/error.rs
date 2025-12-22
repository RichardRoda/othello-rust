use std::fmt;

/// Errors that can occur during gameplay
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameError {
    InvalidMove,
    OutOfBounds,
    GameOver,
    NoValidMoves,
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameError::InvalidMove => write!(f, "Invalid move"),
            GameError::OutOfBounds => write!(f, "Position out of bounds"),
            GameError::GameOver => write!(f, "Game is over"),
            GameError::NoValidMoves => write!(f, "No valid moves available"),
        }
    }
}

impl std::error::Error for GameError {}


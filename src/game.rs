use crate::board::{Board, Position};
use crate::error::GameError;
use crate::rules;

/// Represents a player
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    Black,
    White,
}

impl Player {
    /// Get the opposite player
    pub fn opposite(self) -> Player {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

impl std::fmt::Display for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Player::Black => write!(f, "Black"),
            Player::White => write!(f, "White"),
        }
    }
}

/// Represents the current state of the game
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameState {
    Playing,
    GameOver { winner: Option<Player> },
}

/// Main game state manager
#[derive(Debug, Clone)]
pub struct Game {
    board: Board,
    current_player: Player,
    game_state: GameState,
}

impl Game {
    /// Create a new game with initial setup
    pub fn new() -> Self {
        Game {
            board: Board::new(),
            current_player: Player::Black,
            game_state: GameState::Playing,
        }
    }

    /// Get the current player
    pub fn current_player(&self) -> Player {
        self.current_player
    }

    /// Get the current game state
    pub fn get_game_state(&self) -> GameState {
        self.game_state
    }

    /// Get a reference to the board (read-only)
    pub fn get_board(&self) -> &Board {
        &self.board
    }

    /// Get the current score (black, white)
    pub fn get_score(&self) -> (usize, usize) {
        (
            self.board.count_pieces(Player::Black),
            self.board.count_pieces(Player::White),
        )
    }

    /// Attempt to make a move at the given position
    pub fn make_move(&mut self, position: Position) -> Result<(), GameError> {
        self.make_move_with_flipped(position).map(|_| ())
    }

    /// Attempt to make a move and return the flipped positions (useful for animations)
    pub fn make_move_with_flipped(&mut self, position: Position) -> Result<Vec<Position>, GameError> {
        if self.game_state != GameState::Playing {
            return Err(GameError::GameOver);
        }

        // Validate the move
        if !rules::is_valid_move(&self.board, position, self.current_player) {
            return Err(GameError::InvalidMove);
        }

        // Apply the move (this will flip pieces and return flipped positions)
        let flipped = rules::apply_move(&mut self.board, position, self.current_player)?;

        // Switch to the next player
        self.current_player = self.current_player.opposite();

        // Check if the new current player has valid moves
        if !rules::has_valid_move(&self.board, self.current_player) {
            // Current player can't move, switch back
            self.current_player = self.current_player.opposite();
            
            // Check if this player also can't move (game over)
            if !rules::has_valid_move(&self.board, self.current_player) {
                self.game_state = GameState::GameOver {
                    winner: rules::get_winner(&self.board),
                };
            }
        }

        Ok(flipped)
    }

    /// Skip the current player's turn (called when no valid moves)
    pub fn skip_turn(&mut self) -> Result<(), GameError> {
        if self.game_state != GameState::Playing {
            return Err(GameError::GameOver);
        }

        // Switch to the next player
        self.current_player = self.current_player.opposite();

        // Check if this player also can't move (game over)
        if !rules::has_valid_move(&self.board, self.current_player) {
            self.game_state = GameState::GameOver {
                winner: rules::get_winner(&self.board),
            };
        }

        Ok(())
    }

    /// Get all valid moves for the current player
    pub fn get_valid_moves(&self) -> Vec<Position> {
        rules::get_valid_moves(&self.board, self.current_player)
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}


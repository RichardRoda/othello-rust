use crate::board::Position;
use crate::game::Game;

/// Trait for players (human or AI)
pub trait PlayerTrait {
    /// Choose a move for the current game state
    /// Returns None if no move is available or player wants to quit
    fn choose_move(&self, game: &Game) -> Option<Position>;
    
    /// Get the player's name
    fn get_name(&self) -> &str;
}



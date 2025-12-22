use crate::board::Position;
use crate::game::Game;
use crate::player::PlayerTrait;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// AI player that makes random valid moves
pub struct AIPlayer {
    name: String,
}

impl AIPlayer {
    /// Create a new AI player
    pub fn new(name: impl Into<String>) -> Self {
        AIPlayer {
            name: name.into(),
        }
    }
}

impl PlayerTrait for AIPlayer {
    fn choose_move(&self, game: &Game) -> Option<Position> {
        let valid_moves = game.get_valid_moves();
        
        if valid_moves.is_empty() {
            return None;
        }

        // Choose a random valid move
        let mut rng = thread_rng();
        valid_moves.choose(&mut rng).copied()
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}



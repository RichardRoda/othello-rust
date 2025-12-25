pub mod board;
pub mod error;
pub mod game;
pub mod rules;
pub mod player;
pub mod human_player;
pub mod ai_player;
pub mod display;
pub mod graphics;
pub mod mcts;
pub mod player_selection;
pub mod console_selection;
pub mod gui_selection;

// Re-export commonly used types
pub use board::{Board, Cell, Position};
pub use error::GameError;
pub use game::{Game, GameState, Player};
pub use player::PlayerTrait;
pub use human_player::HumanPlayer;
pub use ai_player::AIPlayer;
pub use graphics::GraphicsState;
pub use mcts::MCTSPlayer;
pub use player_selection::{PlayerType, PlayerConfig};


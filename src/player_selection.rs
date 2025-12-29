use crate::player::PlayerTrait;
use crate::{HumanPlayer, AIPlayer};
use crate::mcts::MCTSPlayer;
use crate::minimax::MinimaxPlayer;

/// Represents the type of player that can be selected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    AIRandom,
    MCTSEasy,
    MCTSMedium,
    MCTSHard,
    MCTSExpert,
    MinimaxEasy,
    MinimaxMedium,
    MinimaxHard,
    MinimaxExpert,
}

impl PlayerType {
    /// Get all available player types
    pub fn all() -> Vec<PlayerType> {
        vec![
            PlayerType::Human,
            PlayerType::AIRandom,
            PlayerType::MinimaxEasy,
            PlayerType::MinimaxMedium,
            PlayerType::MinimaxHard,
            PlayerType::MinimaxExpert,
            PlayerType::MCTSEasy,
            PlayerType::MCTSMedium,
            PlayerType::MCTSHard,
            PlayerType::MCTSExpert,
        ]
    }
    
    /// Get display name for the player type
    pub fn display_name(&self) -> &'static str {
        match self {
            PlayerType::Human => "Human",
            PlayerType::AIRandom => "AI (Random)",
            PlayerType::MCTSEasy => "MCTS (Easy)",
            PlayerType::MCTSMedium => "MCTS (Medium)",
            PlayerType::MCTSHard => "MCTS (Hard)",
            PlayerType::MCTSExpert => "MCTS (Expert)",
            PlayerType::MinimaxEasy => "Minimax (Easy)",
            PlayerType::MinimaxMedium => "Minimax (Medium)",
            PlayerType::MinimaxHard => "Minimax (Hard)",
            PlayerType::MinimaxExpert => "Minimax (Expert)",
        }
    }
    
    /// Create a player instance from this type
    pub fn create_player(&self, name: String) -> Box<dyn PlayerTrait> {
        match self {
            PlayerType::Human => Box::new(HumanPlayer::new(name)),
            PlayerType::AIRandom => Box::new(AIPlayer::new(name)),
            PlayerType::MCTSEasy => Box::new(MCTSPlayer::easy()),
            PlayerType::MCTSMedium => Box::new(MCTSPlayer::medium()),
            PlayerType::MCTSHard => Box::new(MCTSPlayer::hard()),
            PlayerType::MCTSExpert => Box::new(MCTSPlayer::expert()),
            PlayerType::MinimaxEasy => Box::new(MinimaxPlayer::easy()),
            PlayerType::MinimaxMedium => Box::new(MinimaxPlayer::medium()),
            PlayerType::MinimaxHard => Box::new(MinimaxPlayer::hard()),
            PlayerType::MinimaxExpert => Box::new(MinimaxPlayer::expert()),
        }
    }
    
    /// Check if this player type is human
    pub fn is_human(&self) -> bool {
        matches!(self, PlayerType::Human)
    }
}

/// Configuration for both players
#[derive(Debug, Clone)]
pub struct PlayerConfig {
    pub black_type: PlayerType,
    pub white_type: PlayerType,
    pub black_name: String,
    pub white_name: String,
}

impl PlayerConfig {
    pub fn new(black_type: PlayerType, white_type: PlayerType) -> Self {
        Self {
            black_type,
            white_type,
            black_name: "Black".to_string(),
            white_name: "White".to_string(),
        }
    }
    
    /// Create player instances from configuration
    pub fn create_players(&self) -> (Box<dyn PlayerTrait>, Box<dyn PlayerTrait>) {
        let black = self.black_type.create_player(self.black_name.clone());
        let white = self.white_type.create_player(self.white_name.clone());
        (black, white)
    }
}


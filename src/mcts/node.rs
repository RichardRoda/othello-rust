use crate::game::{Game, Player, GameState};
use crate::board::Position;

/// A node in the Monte Carlo Tree
pub struct MCTSNode {
    /// Number of times this node has been visited
    visits: usize,
    
    /// Total accumulated value (win rate from current player's perspective)
    value: f64,
    
    /// The game state at this node
    game_state: Game,
    
    /// The move that led to this node (None for root)
    move_from_parent: Option<Position>,
    
    /// Child nodes (one per valid move)
    children: Vec<Box<MCTSNode>>,
    
    /// Whether children have been expanded
    is_expanded: bool,
    
    /// Player whose turn it is at this node
    current_player: Player,
    
    /// Whether this is a terminal (game over) node
    is_terminal: bool,
}

impl MCTSNode {
    /// Create a new root node from a game state
    pub fn new(game_state: Game) -> Self {
        let current_player = game_state.current_player();
        let is_terminal = matches!(game_state.get_game_state(), GameState::GameOver { .. });
        
        MCTSNode {
            visits: 0,
            value: 0.0,
            game_state,
            move_from_parent: None,
            children: Vec::new(),
            is_expanded: false,
            current_player,
            is_terminal,
        }
    }
    
    /// Get the number of visits
    pub fn visits(&self) -> usize {
        self.visits
    }
    
    /// Get the average value (win rate)
    pub fn average_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value / self.visits as f64
        }
    }
    
    /// Check if node is terminal
    pub fn is_terminal(&self) -> bool {
        self.is_terminal
    }
    
    /// Check if node is expanded
    pub fn is_expanded(&self) -> bool {
        self.is_expanded
    }
    
    /// Get the move from parent
    pub fn move_from_parent(&self) -> Option<Position> {
        self.move_from_parent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_creation() {
        let game = Game::new();
        let node = MCTSNode::new(game);
        
        assert_eq!(node.visits(), 0);
        assert_eq!(node.average_value(), 0.0);
        assert!(!node.is_terminal());
        assert!(!node.is_expanded());
        assert_eq!(node.move_from_parent(), None);
    }
    
    #[test]
    fn test_node_with_terminal_game() {
        // Create a game and play it to completion
        // Then create a node and verify is_terminal is true
        // (This test can be added later when we have game manipulation)
    }
}


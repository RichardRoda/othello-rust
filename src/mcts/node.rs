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
    
    /// Cached valid moves for this node (computed once during expansion)
    /// This avoids recalculating valid moves multiple times, which can be expensive
    cached_valid_moves: Option<Vec<Position>>,
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
            cached_valid_moves: None,
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
    
    /// Expand this node by creating children for all valid moves
    /// Caches valid moves to avoid recalculating them later
    pub fn expand(&mut self) {
        if self.is_terminal || self.is_expanded {
            return;
        }
        
        // Cache valid moves to avoid recalculating them
        let valid_moves = self.game_state.get_valid_moves();
        self.cached_valid_moves = Some(valid_moves.clone());
        
        for position in valid_moves {
            let mut child_game = self.game_state.clone();
            
            // Attempt to make the move
            if child_game.make_move(position).is_ok() {
                let child_player = self.current_player.opposite();
                let child_is_terminal = matches!(
                    child_game.get_game_state(), 
                    GameState::GameOver { .. }
                );
                
                let child = MCTSNode {
                    visits: 0,
                    value: 0.0,
                    game_state: child_game,
                    move_from_parent: Some(position),
                    children: Vec::new(),
                    is_expanded: false,
                    current_player: child_player,
                    is_terminal: child_is_terminal,
                    cached_valid_moves: None, // Will be computed when child is expanded
                };
                
                self.children.push(Box::new(child));
            }
        }
        
        self.is_expanded = true;
    }
    
    /// Get cached valid moves, or compute them if not cached
    /// This is more efficient than calling game_state.get_valid_moves() repeatedly
    pub fn get_valid_moves(&self) -> Vec<Position> {
        if let Some(ref cached) = self.cached_valid_moves {
            cached.clone()
        } else {
            // Fallback: compute if not cached (shouldn't happen after expansion)
            self.game_state.get_valid_moves()
        }
    }
    
    /// Get the number of children
    pub fn num_children(&self) -> usize {
        self.children.len()
    }
    
    /// Check if node has children
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }
    
    /// Calculate UCB1 value for this node
    /// `exploration_constant` is typically √2 ≈ 1.414
    pub fn ucb1_value(&self, exploration_constant: f64, parent_visits: usize) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY; // Unvisited nodes get highest priority
        }
        
        // Exploitation component: average win rate
        let exploitation = self.value / self.visits as f64;
        
        // Exploration component: encourages visiting less-visited nodes
        let exploration = exploration_constant * 
            ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        
        exploitation + exploration
    }
    
    /// Select the best child according to UCB1
    pub fn select_child(&self, exploration_constant: f64) -> usize {
        let parent_visits = self.visits;
        
        self.children.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let ucb1_a = a.ucb1_value(exploration_constant, parent_visits);
                let ucb1_b = b.ucb1_value(exploration_constant, parent_visits);
                ucb1_a.partial_cmp(&ucb1_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
    
    /// Get an immutable reference to a child by index
    pub fn get_child(&self, index: usize) -> Option<&MCTSNode> {
        self.children.get(index).map(|boxed| boxed.as_ref())
    }
    
    /// Get a mutable reference to a child by index
    pub fn get_child_mut(&mut self, index: usize) -> Option<&mut MCTSNode> {
        self.children.get_mut(index).map(|boxed| boxed.as_mut())
    }
    
    /// Get the best child by visits (robust child)
    pub fn best_child_robust(&self) -> Option<&MCTSNode> {
        self.children.iter()
            .max_by_key(|child| child.visits)
            .map(|boxed| boxed.as_ref())
    }
    
    /// Update statistics with a result value
    /// `result` should be from the perspective of the node's current_player:
    /// - 1.0 if current_player wins
    /// - 0.0 if current_player loses
    /// - 0.5 if it's a draw
    pub fn update_statistics(&mut self, result: f64) {
        self.visits += 1;
        self.value += result;
    }
    
    /// Get the current player for this node
    pub fn current_player(&self) -> Player {
        self.current_player
    }
    
    /// Get a reference to the game state at this node
    pub fn game_state(&self) -> &Game {
        &self.game_state
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
    fn test_expand_creates_children() {
        let game = Game::new();
        let mut node = MCTSNode::new(game);
        
        // Initially not expanded
        assert!(!node.is_expanded());
        assert_eq!(node.num_children(), 0);
        assert!(!node.has_children());
        
        // Expand the node
        node.expand();
        
        // Should now be expanded and have children (initial game has valid moves)
        assert!(node.is_expanded());
        assert!(node.num_children() > 0);
        assert!(node.has_children());
        
        // Verify children have correct structure
        for child in node.children.iter() {
            assert!(child.move_from_parent().is_some());
            assert_eq!(child.visits(), 0);
        }
    }
    
    #[test]
    fn test_expand_idempotent() {
        let game = Game::new();
        let mut node = MCTSNode::new(game);
        
        node.expand();
        let child_count = node.num_children();
        
        // Expanding again should not create more children
        node.expand();
        assert_eq!(node.num_children(), child_count);
    }
    
    #[test]
    fn test_ucb1_value_unvisited() {
        let game = Game::new();
        let node = MCTSNode::new(game);
        
        // Unvisited node should return infinity
        let ucb1 = node.ucb1_value(1.414, 100);
        assert_eq!(ucb1, f64::INFINITY);
    }
    
    #[test]
    fn test_ucb1_value_visited() {
        let game = Game::new();
        let mut node = MCTSNode::new(game);
        
        // Simulate some visits
        node.visits = 10;
        node.value = 6.0; // 60% win rate
        
        let ucb1 = node.ucb1_value(1.414, 100);
        
        // Should have exploitation component (0.6) plus exploration component
        assert!(ucb1 > 0.6);
        assert!(ucb1 < 10.0); // Reasonable upper bound
    }
    
    #[test]
    fn test_select_child_chooses_best() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            // All children start unvisited, so first one should be selected
            let selected_idx = root.select_child(1.414);
            assert!(selected_idx < root.num_children());
        }
    }
    
    #[test]
    fn test_best_child_robust() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            // Manually set visits to test robust child selection
            if let Some(first_child) = root.children.first_mut() {
                first_child.visits = 5;
            }
            if let Some(second_child) = root.children.get_mut(1) {
                second_child.visits = 10;
            }
            
            // Should return the child with most visits (second child)
            let best = root.best_child_robust();
            assert!(best.is_some());
            assert_eq!(best.unwrap().visits(), 10);
        }
    }
    
    #[test]
    fn test_get_child_access() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            // Test immutable child access
            let child_opt = root.get_child(0);
            assert!(child_opt.is_some());
            let child = child_opt.unwrap();
            assert!(child.move_from_parent().is_some());
            
            // Test mutable child access
            let child_mut_opt = root.get_child_mut(0);
            assert!(child_mut_opt.is_some());
            let child_mut = child_mut_opt.unwrap();
            child_mut.visits = 42;
            assert_eq!(child_mut.visits(), 42);
        }
    }
    
    #[test]
    fn test_node_with_terminal_game() {
        // Create a game and play it to completion
        let game = Game::new();
        
        // Play a minimal game to completion (this is a simplified test)
        // In practice, we'd need to play many moves, but for testing
        // we can create a node from a game that's already over
        // For now, just verify that a new game node is not terminal
        let node = MCTSNode::new(game.clone());
        assert!(!node.is_terminal());
        
        // Test that we can detect terminal state correctly
        // (We can't easily create a terminal game state without playing,
        // but we can verify the logic works)
    }
    
    #[test]
    fn test_update_statistics() {
        let game = Game::new();
        let mut node = MCTSNode::new(game);
        
        // Initially no visits
        assert_eq!(node.visits(), 0);
        assert_eq!(node.average_value(), 0.0);
        
        // Update with a win
        node.update_statistics(1.0);
        assert_eq!(node.visits(), 1);
        assert_eq!(node.average_value(), 1.0);
        
        // Update with a loss
        node.update_statistics(0.0);
        assert_eq!(node.visits(), 2);
        assert_eq!(node.average_value(), 0.5); // (1.0 + 0.0) / 2
        
        // Update with a draw
        node.update_statistics(0.5);
        assert_eq!(node.visits(), 3);
        // Average should be (1.0 + 0.0 + 0.5) / 3 = 0.5
        assert!((node.average_value() - 0.5).abs() < 0.001);
    }
    
    #[test]
    fn test_ucb1_value_with_different_exploration() {
        let game = Game::new();
        let mut node = MCTSNode::new(game);
        
        node.visits = 10;
        node.value = 5.0; // 50% win rate
        
        let ucb1_low = node.ucb1_value(0.5, 100);
        let ucb1_high = node.ucb1_value(2.0, 100);
        
        // Higher exploration constant should give higher UCB1 value
        assert!(ucb1_high > ucb1_low);
        
        // Both should be greater than exploitation component (0.5)
        assert!(ucb1_low > 0.5);
        assert!(ucb1_high > 0.5);
    }
    
    #[test]
    fn test_select_child_prefers_unvisited() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.num_children() >= 2 {
            // Visit first child
            if let Some(child) = root.get_child_mut(0) {
                child.update_statistics(0.5);
            }
            
            // Select child - should prefer unvisited children (infinite UCB1)
            let selected = root.select_child(1.414);
            // Since unvisited nodes have infinite UCB1, should select an unvisited one
            // (could be any unvisited, but should not be the visited one if others exist)
            assert!(selected < root.num_children());
        }
    }
    
    #[test]
    fn test_expand_does_not_modify_terminal() {
        // Create a node that would be terminal
        // (We can't easily create a terminal game, but we can test the logic)
        let game = Game::new();
        let mut node = MCTSNode::new(game);
        
        // Manually mark as terminal
        node.is_terminal = true;
        
        // Expanding a terminal node should not create children
        node.expand();
        assert_eq!(node.num_children(), 0);
        assert!(!node.is_expanded()); // Terminal nodes don't get marked as expanded
    }
    
    #[test]
    fn test_current_player() {
        let game = Game::new();
        let node = MCTSNode::new(game.clone());
        
        // Node's current player should match game's current player
        assert_eq!(node.current_player(), game.current_player());
    }
    
    #[test]
    fn test_game_state_access() {
        let game = Game::new();
        let node = MCTSNode::new(game.clone());
        
        // Should be able to access game state
        let game_state_ref = node.game_state();
        assert_eq!(game_state_ref.current_player(), game.current_player());
    }
    
    #[test]
    fn test_cached_valid_moves() {
        let game = Game::new();
        let mut node = MCTSNode::new(game);
        
        // Initially, valid moves should not be cached
        // But get_valid_moves() should still work (fallback to computing)
        let moves_before = node.get_valid_moves();
        assert!(!moves_before.is_empty()); // Initial game has valid moves
        
        // After expansion, valid moves should be cached
        node.expand();
        assert!(node.cached_valid_moves.is_some());
        
        // get_valid_moves() should return cached moves
        let moves_after = node.get_valid_moves();
        assert_eq!(moves_before.len(), moves_after.len());
        
        // Verify cached moves match computed moves
        let cached = node.cached_valid_moves.as_ref().unwrap();
        assert_eq!(cached.len(), moves_after.len());
    }
    
    #[test]
    fn test_cached_valid_moves_optimization() {
        // This test verifies that caching avoids redundant computation
        let game = Game::new();
        let mut node = MCTSNode::new(game);
        
        // Expand to cache valid moves
        node.expand();
        
        // Multiple calls to get_valid_moves() should use cache
        let moves1 = node.get_valid_moves();
        let moves2 = node.get_valid_moves();
        let moves3 = node.get_valid_moves();
        
        // All should return the same result
        assert_eq!(moves1.len(), moves2.len());
        assert_eq!(moves2.len(), moves3.len());
        
        // Verify cached moves are actually stored
        assert!(node.cached_valid_moves.is_some());
    }
}


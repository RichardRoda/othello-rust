# MCTS Implementation Strategy

This document provides a detailed, step-by-step implementation guide for adding Monte Carlo Tree Search (MCTS) to the Othello game. It translates the design from `MCTS_DESIGN.md` into concrete, actionable tasks.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Phase 1: Module Setup and Node Structure](#phase-1-module-setup-and-node-structure)
3. [Phase 2: Core MCTS Algorithm](#phase-2-core-mcts-algorithm)
4. [Phase 3: Selection and Expansion](#phase-3-selection-and-expansion)
5. [Phase 4: Simulation and Backpropagation](#phase-4-simulation-and-backpropagation)
6. [Phase 5: Player Integration](#phase-5-player-integration)
7. [Phase 6: Heuristics](#phase-6-heuristics)
8. [Phase 7: Testing and Validation](#phase-7-testing-and-validation)
9. [Phase 8: Performance Optimization](#phase-8-performance-optimization)
10. [Phase 9: Polish and Documentation](#phase-9-polish-and-documentation)

---

## Prerequisites

Before starting implementation, ensure you have:

- ✅ Rust toolchain installed and up to date
- ✅ Project compiles successfully (`cargo build`)
- ✅ Understanding of existing codebase structure
- ✅ Familiarity with MCTS algorithm concepts
- ✅ Test framework ready (Rust's built-in `#[test]`)

### Codebase Review Checklist

Review these files to understand the existing structure:

- [ ] `src/game.rs` - Understand `Game` struct and methods
- [ ] `src/board.rs` - Understand `Board` and `Position`
- [ ] `src/player.rs` - Understand `PlayerTrait` interface
- [ ] `src/ai_player.rs` - See existing AI player implementation
- [ ] `src/rules.rs` - Understand move validation and game rules
- [ ] `src/lib.rs` - See module organization

### Key Methods to Use

```rust
// From Game:
game.clone()                    // Clone game state
game.current_player()           // Get current player
game.get_valid_moves()          // Get Vec<Position>
game.make_move(position)        // Returns Result<(), GameError>
game.skip_turn()                // Returns Result<(), GameError>
game.get_game_state()           // Returns GameState
game.get_board()                // Returns &Board
game.get_score()                // Returns (black_count, white_count)

// From Player:
player.opposite()               // Get opposite player

// From Position:
Position::new(row, col)         // Create position

// From GameState:
GameState::Playing              // Still playing
GameState::GameOver { winner }  // Game ended
```

---

## Phase 1: Module Setup and Node Structure

**Estimated Time:** 2-3 hours  
**Goal:** Create the module structure and basic `MCTSNode` implementation.

### Step 1.1: Create Module Structure

**Task:** Create the `mcts` module directory and files.

```bash
mkdir src/mcts
touch src/mcts/mod.rs
touch src/mcts/node.rs
touch src/mcts/player.rs
```

**File: `src/mcts/mod.rs`**
```rust
pub mod node;
pub mod player;

// Re-export for convenience
pub use player::MCTSPlayer;
```

**Action Items:**
- [ ] Create `src/mcts/` directory
- [ ] Create `mod.rs`, `node.rs`, `player.rs` files
- [ ] Add `pub mod mcts;` to `src/lib.rs`
- [ ] Verify `cargo build` works (will have errors, that's OK)

### Step 1.2: Implement Basic MCTSNode Structure

**Task:** Create the `MCTSNode` struct with all required fields.

**File: `src/mcts/node.rs`**
```rust
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
```

**Action Items:**
- [ ] Implement `MCTSNode` struct with all fields
- [ ] Implement `new()` constructor
- [ ] Add getter methods
- [ ] Run `cargo build` to check for compilation errors
- [ ] Fix any import or type errors

### Step 1.3: Write Basic Tests

**Task:** Write tests to verify node creation works correctly.

**Add to `src/mcts/node.rs`:**
```rust
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
```

**Action Items:**
- [ ] Add basic test for node creation
- [ ] Run `cargo test` to verify tests pass
- [ ] Commit: "Phase 1: Add MCTS module structure and basic node"

---

## Phase 2: Core MCTS Algorithm

**Estimated Time:** 4-6 hours  
**Goal:** Implement the main MCTS search loop and basic tree operations.

### Step 2.1: Implement Expansion

**Task:** Add logic to expand a node by creating child nodes for all valid moves.

**Add to `src/mcts/node.rs`:**
```rust
impl MCTSNode {
    /// Expand this node by creating children for all valid moves
    pub fn expand(&mut self) {
        if self.is_terminal || self.is_expanded {
            return;
        }
        
        let valid_moves = self.game_state.get_valid_moves();
        
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
                };
                
                self.children.push(Box::new(child));
            }
        }
        
        self.is_expanded = true;
    }
    
    /// Get the number of children
    pub fn num_children(&self) -> usize {
        self.children.len()
    }
    
    /// Check if node has children
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }
}
```

**Action Items:**
- [ ] Implement `expand()` method
- [ ] Add helper methods (`num_children`, `has_children`)
- [ ] Test expansion with `cargo test`
- [ ] Verify children are created correctly for initial game state

### Step 2.2: Implement UCB1 Selection

**Task:** Implement the UCB1 formula for selecting child nodes.

**Add to `src/mcts/node.rs`:**
```rust
impl MCTSNode {
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
    
    /// Get a mutable reference to a child by index
    pub fn get_child_mut(&mut self, index: usize) -> Option<&mut MCTSNode> {
        self.children.get_mut(index).map(|boxed| boxed.as_mut())
    }
    
    /// Get the best child by visits (robust child)
    pub fn best_child_robust(&self) -> Option<&MCTSNode> {
        self.children.iter()
            .max_by_key(|child| child.visits)
    }
}
```

**Action Items:**
- [ ] Implement `ucb1_value()` method
- [ ] Implement `select_child()` method
- [ ] Implement `get_child_mut()` helper
- [ ] Implement `best_child_robust()` for final move selection
- [ ] Write tests for UCB1 calculation
- [ ] Test selection with multiple children

### Step 2.3: Implement Tree Traversal

**Task:** Implement selection phase that traverses from root to leaf.

**Add to `src/mcts/player.rs` (create skeleton for now):**
```rust
use crate::game::Game;
use crate::board::Position;
use crate::mcts::node::MCTSNode;
use std::time::Instant;

pub struct MCTSPlayer {
    name: String,
    iterations: usize,
    exploration_constant: f64,
    max_time_ms: Option<u64>,
    use_heuristics: bool,
}

impl MCTSPlayer {
    /// Perform selection phase: traverse from root to leaf (read-only to get path)
    fn select_path(root: &MCTSNode, exploration_constant: f64) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = root;
        
        // Traverse until we find an unexpanded or terminal node
        while current.is_expanded() && !current.is_terminal() && current.has_children() {
            let child_idx = current.select_child(exploration_constant);
            path.push(child_idx);
            // Get child reference (immutable)
            current = &current.children[child_idx];
        }
        
        path
    }
    
    /// Get mutable reference to node at path
    fn get_node_mut_at_path(root: &mut MCTSNode, path: &[usize]) -> &mut MCTSNode {
        let mut current = root;
        for &idx in path {
            current = current.get_child_mut(idx)
                .expect("Path index should be valid");
        }
        current
    }
}
```

**Action Items:**
- [ ] Create `MCTSPlayer` struct skeleton
- [ ] Implement `select_leaf()` method
- [ ] Implement `get_node_at_path()` helper
- [ ] Test tree traversal with simple scenarios
- [ ] Handle edge cases (terminal nodes, empty children)

---

## Phase 3: Selection and Expansion

**Estimated Time:** 3-4 hours  
**Goal:** Complete the selection and expansion phases of MCTS.

### Step 3.1: Refine Selection Logic

**Task:** Improve selection to handle all edge cases properly.

**Update `src/mcts/player.rs`:**
```rust
impl MCTSPlayer {
    /// Find a leaf node through selection, return path indices
    /// This is a read-only operation that returns a path
    fn select_path(root: &MCTSNode, exploration_constant: f64) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = root;
        
        // Keep traversing while node is expanded, not terminal, and has children
        while current.is_expanded() && !current.is_terminal() && current.has_children() {
            let child_idx = current.select_child(exploration_constant);
            path.push(child_idx);
            
            // Move to child (immutable reference)
            current = &current.children[child_idx];
        }
        
        path
    }
}
```

**Issue:** Rust's ownership system makes tree traversal tricky. We need a different approach.

**Solution:** Use recursion or restructure to work with tree by index.

**Action Items:**
- [ ] Research tree traversal patterns in Rust
- [ ] Decide on traversal approach (recursive vs iterative)
- [ ] Implement proper selection that handles borrowing
- [ ] Test with multi-level trees

### Step 3.2: Implement Mutable Tree Traversal

**Task:** Implement a way to traverse and modify tree nodes.

**Better approach - use a helper struct:**
```rust
// In src/mcts/node.rs

impl MCTSNode {
    /// Traverse to leaf and return reference (for selection only)
    fn find_leaf_for_expansion(&mut self) -> &mut MCTSNode {
        let mut current = self;
        
        while current.is_expanded && !current.is_terminal && current.has_children() {
            let child_idx = current.select_child(1.414); // Use default for now
            // This won't work directly due to borrowing...
        }
        
        current
    }
}
```

**Solution: Separate selection (read-only) from modification**
```rust
// This approach uses immutable references for selection,
// then uses the path to get mutable references when needed
impl MCTSPlayer {
    fn select_path(root: &MCTSNode, exploration_constant: f64) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = root;
        
        while current.is_expanded() && !current.is_terminal() && current.has_children() {
            let idx = current.select_child(exploration_constant);
            path.push(idx);
            current = &current.children[idx];
        }
        
        path
    }
    
    fn get_node_mut_at_path(root: &mut MCTSNode, path: &[usize]) -> &mut MCTSNode {
        let mut current = root;
        for &idx in path {
            current = current.get_child_mut(idx).expect("Path index should be valid");
        }
        current
    }
}
```

**Action Items:**
- [ ] Implement `select_path()` (read-only)
- [ ] Implement `get_node_mut_at_path()` helper
- [ ] Test path selection and node retrieval
- [ ] Verify it works with deep trees

---

## Phase 4: Simulation and Backpropagation

**Estimated Time:** 4-5 hours  
**Goal:** Implement simulation (rollout) and backpropagation phases.

### Step 4.1: Implement Basic Random Simulation

**Task:** Implement random game simulation from a node to terminal state.

**Add to `src/mcts/player.rs`:**
```rust
use rand::seq::SliceRandom;
use rand::thread_rng;

impl MCTSPlayer {
    /// Simulate a random game from given state
    /// Returns: 1.0 if root_player wins, 0.0 if loses, 0.5 if draw
    fn simulate(&self, game: &mut Game, root_player: Player) -> f64 {
        let mut rng = thread_rng();
        
        // Play random moves until game ends
        while matches!(game.get_game_state(), GameState::Playing) {
            let valid_moves = game.get_valid_moves();
            
            if valid_moves.is_empty() {
                // No valid moves, skip turn
                game.skip_turn().ok();
                continue;
            }
            
            // Choose random move
            if let Some(&move_pos) = valid_moves.choose(&mut rng) {
                game.make_move(move_pos).ok();
            }
        }
        
        // Determine result from root player's perspective
        match game.get_game_state() {
            GameState::GameOver { winner } => {
                match winner {
                    Some(player) if player == root_player => 1.0,
                    Some(_) => 0.0,
                    None => 0.5,
                }
            }
            _ => 0.5, // Should not happen
        }
    }
}
```

**Action Items:**
- [ ] Add `rand` dependency if not already present
- [ ] Implement `simulate()` method
- [ ] Test that simulation always reaches terminal state
- [ ] Verify result calculation is correct
- [ ] Add test for simulation outcome

### Step 4.2: Implement Backpropagation

**Task:** Update node statistics along the path from leaf to root.

**Add to `src/mcts/player.rs`:**
```rust
impl MCTSPlayer {
    /// Backpropagate result up the tree
    fn backpropagate(root: &mut MCTSNode, path: &[usize], result: f64, root_player: Player) {
        // Update all nodes along the path, including root
        // We need to update in reverse order to handle borrowing
        
        // Update leaf first
        let leaf = Self::get_node_mut_at_path(root, path);
        let leaf_player = leaf.current_player;
        let leaf_result = if leaf_player == root_player { result } else { 1.0 - result };
        leaf.visits += 1;
        leaf.value += leaf_result;
        
        // Now update nodes along path (working backwards from leaf to root)
        // We can't do this in one pass due to borrowing, so we update root separately
    }
    
    // Helper to update root separately
    fn update_root(root: &mut MCTSNode, result: f64) {
        root.visits += 1;
        root.value += result;
    }
    
    // Helper to update nodes along path
    fn update_path_nodes(root: &mut MCTSNode, path: &[usize], result: f64, root_player: Player) {
        // Update each node along the path
        for i in 0..path.len() {
            let node_path = &path[..=i];  // Path up to and including this node
            let node = Self::get_node_mut_at_path(root, node_path);
            let node_player = node.current_player;
            let node_result = if node_player == root_player { result } else { 1.0 - result };
            node.visits += 1;
            node.value += node_result;
        }
    }
}
```

**Wait - there's an issue with the result flipping logic. Let's fix it:**

**Better implementation:**
```rust
impl MCTSPlayer {
    fn backpropagate(root: &mut MCTSNode, path: &[usize], result: f64, root_player: Player) {
        // Update root first (separate to avoid borrowing issues)
        root.visits += 1;
        root.value += result;
        
        // Update nodes along path
        for i in 0..path.len() {
            let node_path = &path[..=i];  // Path up to this node (inclusive)
            let node = Self::get_node_mut_at_path(root, node_path);
            let node_player = node.current_player;
            // Result from this node's player perspective
            let node_result = if node_player == root_player { result } else { 1.0 - result };
            node.visits += 1;
            node.value += node_result;
        }
    }
}
```

**Action Items:**
- [ ] Implement `backpropagate()` method
- [ ] Test that statistics update correctly
- [ ] Verify result flipping works for alternating players
- [ ] Test with single-level and multi-level paths
- [ ] Add assertions to verify visits and values are correct

### Step 4.3: Integrate All Phases

**Task:** Combine selection, expansion, simulation, and backpropagation.

**Add to `src/mcts/player.rs`:**
```rust
impl MCTSPlayer {
    /// Perform one MCTS iteration
    fn mcts_iteration(&self, root: &mut MCTSNode, root_player: Player) {
        // 1. Selection: Find path to leaf (read-only)
        let path = Self::select_path(root, self.exploration_constant);
        
        // 2. Get the leaf node (mutable)
        let leaf = Self::get_node_mut_at_path(root, &path);
        
        // 3. Expansion: Expand if not terminal and not expanded
        if !leaf.is_terminal() && !leaf.is_expanded() {
            leaf.expand();
        }
        
        // 4. Simulation: Play random game from leaf state
        let mut sim_game = leaf.game_state.clone();
        let result = self.simulate(&mut sim_game, root_player);
        
        // 5. Backpropagation: Update statistics along path
        Self::backpropagate(root, &path, result, root_player);
    }
}
```

**Action Items:**
- [ ] Implement `mcts_iteration()` method
- [ ] Test complete iteration cycle
- [ ] Verify tree grows correctly
- [ ] Test multiple iterations
- [ ] Check that statistics accumulate properly

---

## Phase 5: Player Integration

**Estimated Time:** 3-4 hours  
**Goal:** Implement `PlayerTrait` for `MCTSPlayer` and integrate with game.

### Step 5.1: Complete MCTSPlayer Implementation

**Task:** Finish `MCTSPlayer` struct and constructor methods.

**Update `src/mcts/player.rs`:**
```rust
use crate::player::PlayerTrait;
use crate::game::{Game, Player};
use crate::board::Position;
use crate::mcts::node::MCTSNode;
use std::time::Instant;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct MCTSPlayer {
    name: String,
    iterations: usize,
    exploration_constant: f64,
    max_time_ms: Option<u64>,
    use_heuristics: bool,
}

impl MCTSPlayer {
    pub fn new(name: impl Into<String>) -> Self {
        Self::with_iterations(name, 1000)
    }
    
    pub fn with_iterations(name: impl Into<String>, iterations: usize) -> Self {
        Self {
            name: name.into(),
            iterations,
            exploration_constant: 1.414, // √2
            max_time_ms: None,
            use_heuristics: false,
        }
    }
    
    pub fn set_exploration_constant(&mut self, c: f64) {
        self.exploration_constant = c;
    }
    
    pub fn set_max_time_ms(&mut self, ms: Option<u64>) {
        self.max_time_ms = ms;
    }
    
    pub fn set_use_heuristics(&mut self, use: bool) {
        self.use_heuristics = use;
    }
    
    /// Perform MCTS search and return best move
    fn mcts_search(&self, game: &Game) -> Option<Position> {
        let mut root = MCTSNode::new(game.clone());
        let root_player = game.current_player();
        let start_time = Instant::now();
        
        for iteration in 0..self.iterations {
            // Check time limit
            if let Some(max_ms) = self.max_time_ms {
                if start_time.elapsed().as_millis() as u64 > max_ms {
                    break;
                }
            }
            
            // Perform one MCTS iteration
            self.mcts_iteration(&mut root, root_player);
        }
        
        // Return move from most visited child
        root.best_child_robust()
            .and_then(|child| child.move_from_parent())
    }
}
```

**Action Items:**
- [ ] Complete `MCTSPlayer` struct
- [ ] Implement constructor methods
- [ ] Implement `mcts_search()` method
- [ ] Add time limit checking
- [ ] Test with small iteration counts

### Step 5.2: Implement PlayerTrait

**Task:** Make `MCTSPlayer` implement `PlayerTrait`.

**Add to `src/mcts/player.rs`:**
```rust
impl PlayerTrait for MCTSPlayer {
    fn choose_move(&self, game: &Game) -> Option<Position> {
        let valid_moves = game.get_valid_moves();
        
        if valid_moves.is_empty() {
            return None; // No valid moves
        }
        
        // If only one move, return it immediately
        if valid_moves.len() == 1 {
            return Some(valid_moves[0]);
        }
        
        // Perform MCTS search
        self.mcts_search(game)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}
```

**Action Items:**
- [ ] Implement `PlayerTrait` for `MCTSPlayer`
- [ ] Handle edge case: no valid moves
- [ ] Handle edge case: only one valid move
- [ ] Test `choose_move()` with simple game states
- [ ] Verify it returns valid moves

### Step 5.3: Integration Test

**Task:** Test MCTS player in a real game scenario.

**Create `tests/mcts_integration_test.rs`:**
```rust
use othello::*;
use othello::mcts::MCTSPlayer;

#[test]
fn test_mcts_chooses_move() {
    let game = Game::new();
    let player = MCTSPlayer::with_iterations("Test MCTS", 100);
    
    let move_opt = player.choose_move(&game);
    assert!(move_opt.is_some());
    
    // Verify move is valid
    let position = move_opt.unwrap();
    let valid_moves = game.get_valid_moves();
    assert!(valid_moves.contains(&position));
}

#[test]
fn test_mcts_vs_random() {
    // Play a short game: MCTS vs Random
    // MCTS should generally perform better
    // (This is a longer test that can be run separately)
}
```

**Action Items:**
- [ ] Create integration test file
- [ ] Test basic move selection
- [ ] Test with game loop (optional, can defer)
- [ ] Run `cargo test` to verify
- [ ] Commit: "Phase 5: Integrate MCTS player with game"

---

## Phase 6: Heuristics

**Estimated Time:** 5-6 hours  
**Goal:** Add heuristics to improve simulation quality.

### Step 6.1: Create Heuristics Module

**Task:** Create separate module for heuristics.

**File: `src/mcts/heuristics.rs`**
```rust
use crate::game::Game;
use crate::board::Position;

pub struct Heuristics;

impl Heuristics {
    /// Evaluate a move using all heuristics
    pub fn evaluate_move(game: &Game, position: Position) -> f64 {
        let corner_value = Self::corner_heuristic(position);
        let mobility_value = Self::mobility_heuristic(game, position);
        let stability_value = Self::stability_heuristic(position);
        
        // Weighted combination
        corner_value * 10.0 + mobility_value * 2.0 + stability_value * 1.0
    }
    
    /// Corner heuristic: corners are valuable, adjacent to corners is bad
    pub fn corner_heuristic(position: Position) -> f64 {
        let corners = [
            Position::new(0, 0), Position::new(0, 7),
            Position::new(7, 0), Position::new(7, 7),
        ];
        
        if corners.contains(&position) {
            1.0
        } else if Self::is_adjacent_to_corner(position) {
            -0.5
        } else {
            0.0
        }
    }
    
    /// Check if position is adjacent to a corner
    fn is_adjacent_to_corner(position: Position) -> bool {
        let row = position.row;
        let col = position.col;
        
        // Positions adjacent to corners (C-squares and X-squares)
        (row == 0 && (col == 1 || col == 6)) ||
        (row == 7 && (col == 1 || col == 6)) ||
        (col == 0 && (row == 1 || row == 6)) ||
        (col == 7 && (row == 1 || row == 6)) ||
        // X-squares (diagonal to corners)
        (row == 1 && (col == 1 || col == 6)) ||
        (row == 6 && (col == 1 || col == 6))
    }
    
    /// Mobility heuristic: more moves = better
    pub fn mobility_heuristic(game: &Game, position: Position) -> f64 {
        let mut test_game = game.clone();
        if test_game.make_move(position).is_err() {
            return 0.0;
        }
        
        let my_moves = test_game.get_valid_moves().len();
        if test_game.skip_turn().is_err() {
            return my_moves as f64 / 10.0; // Normalize
        }
        
        let opponent_moves = test_game.get_valid_moves().len();
        
        if my_moves + opponent_moves == 0 {
            0.0
        } else {
            (my_moves as f64 - opponent_moves as f64) / (my_moves + opponent_moves) as f64
        }
    }
    
    /// Stability heuristic: edge pieces are more stable
    pub fn stability_heuristic(position: Position) -> f64 {
        let row = position.row;
        let col = position.col;
        
        // Edge bonus
        let edge_bonus = if row == 0 || row == 7 || col == 0 || col == 7 {
            0.3
        } else {
            0.0
        };
        
        // X-squares (adjacent to corners on edge) are risky
        let is_x_square = (row == 0 && (col == 1 || col == 6)) ||
                          (row == 7 && (col == 1 || col == 6)) ||
                          (col == 0 && (row == 1 || row == 6)) ||
                          (col == 7 && (row == 1 || row == 6));
        
        if is_x_square {
            -0.2
        } else {
            edge_bonus
        }
    }
}
```

**Action Items:**
- [ ] Create `src/mcts/heuristics.rs`
- [ ] Add `pub mod heuristics;` to `src/mcts/mod.rs`
- [ ] Implement corner heuristic
- [ ] Implement mobility heuristic
- [ ] Implement stability heuristic
- [ ] Test each heuristic separately
- [ ] Test combined evaluation

### Step 6.2: Integrate Heuristics into Simulation

**Task:** Use heuristics in simulation instead of pure random play.

**Update `src/mcts/player.rs`:**
```rust
use crate::mcts::heuristics::Heuristics;

impl MCTSPlayer {
    fn simulate(&self, game: &mut Game, root_player: Player) -> f64 {
        let mut rng = thread_rng();
        
        while matches!(game.get_game_state(), GameState::Playing) {
            let valid_moves = game.get_valid_moves();
            
            if valid_moves.is_empty() {
                game.skip_turn().ok();
                continue;
            }
            
            // Choose move based on strategy
            let move_pos = if self.use_heuristics {
                self.heuristic_move_selection(game, &valid_moves, &mut rng)
            } else {
                valid_moves.choose(&mut rng).copied()
            };
            
            if let Some(pos) = move_pos {
                game.make_move(pos).ok();
            } else {
                game.skip_turn().ok();
            }
        }
        
        // Determine result...
        match game.get_game_state() {
            GameState::GameOver { winner } => {
                match winner {
                    Some(player) if player == root_player => 1.0,
                    Some(_) => 0.0,
                    None => 0.5,
                }
            }
            _ => 0.5,
        }
    }
    
    fn heuristic_move_selection(
        &self,
        game: &Game,
        moves: &[Position],
        rng: &mut impl rand::Rng,
    ) -> Option<Position> {
        // Score all moves
        let scored_moves: Vec<(Position, f64)> = moves.iter()
            .map(|&pos| {
                let score = Heuristics::evaluate_move(game, pos);
                (pos, score.max(0.01)) // Ensure positive scores for probability
            })
            .collect();
        
        // Calculate total score
        let total_score: f64 = scored_moves.iter().map(|(_, score)| score).sum();
        
        if total_score == 0.0 {
            return moves.choose(rng).copied();
        }
        
        // Weighted random selection
        let mut random_value: f64 = rng.gen();
        random_value *= total_score;
        
        let mut cumulative = 0.0;
        for (pos, score) in scored_moves {
            cumulative += score;
            if random_value <= cumulative {
                return Some(pos);
            }
        }
        
        // Fallback (shouldn't happen)
        moves.last().copied()
    }
}
```

**Action Items:**
- [ ] Import `Heuristics` module
- [ ] Update `simulate()` to use heuristics conditionally
- [ ] Implement `heuristic_move_selection()` method
- [ ] Test heuristic-based simulation
- [ ] Compare results with/without heuristics
- [ ] Tune heuristic weights if needed

---

## Phase 7: Testing and Validation

**Estimated Time:** 4-5 hours  
**Goal:** Comprehensive testing of MCTS implementation.

### Step 7.1: Unit Tests

**Task:** Write comprehensive unit tests for all components.

**Test checklist:**
- [ ] Test `MCTSNode::new()` with various game states
- [ ] Test `expand()` creates correct children
- [ ] Test `ucb1_value()` calculation
- [ ] Test `select_child()` chooses best child
- [ ] Test `simulate()` always reaches terminal state
- [ ] Test `backpropagate()` updates statistics correctly
- [ ] Test heuristics return expected values
- [ ] Test `MCTSPlayer::choose_move()` with edge cases

### Step 7.2: Integration Tests

**Task:** Test MCTS in full game scenarios.

**Create `tests/mcts_game_test.rs`:**
```rust
use othello::*;
use othello::mcts::MCTSPlayer;
use othello::ai_player::AIPlayer;

#[test]
fn test_mcts_completes_game() {
    let mut game = Game::new();
    let player1 = MCTSPlayer::with_iterations("MCTS 1", 50);
    let player2 = MCTSPlayer::with_iterations("MCTS 2", 50);
    
    let mut move_count = 0;
    while matches!(game.get_game_state(), GameState::Playing) && move_count < 100 {
        let player = match game.current_player() {
            Player::Black => &player1 as &dyn PlayerTrait,
            Player::White => &player2 as &dyn PlayerTrait,
        };
        
        if let Some(position) = player.choose_move(&game) {
            game.make_move(position).unwrap();
            move_count += 1;
        } else {
            game.skip_turn().unwrap();
        }
    }
    
    // Game should have ended
    assert!(matches!(game.get_game_state(), GameState::GameOver { .. }));
}

#[test]
fn test_mcts_beats_random() {
    // Play 10 games: MCTS vs Random
    // MCTS should win most games
    let mut mcts_wins = 0;
    
    for _ in 0..10 {
        let mut game = Game::new();
        let mcts = MCTSPlayer::with_iterations("MCTS", 200);
        let random = AIPlayer::new("Random");
        
        // Play game...
        // (Implementation similar to above)
        
        // Check winner
        match game.get_game_state() {
            GameState::GameOver { winner } => {
                if winner == Some(Player::Black) {
                    mcts_wins += 1;
                }
            }
            _ => {}
        }
    }
    
    // MCTS should win at least 60% of games
    assert!(mcts_wins >= 6, "MCTS won {}/10 games", mcts_wins);
}
```

**Action Items:**
- [ ] Create integration test file
- [ ] Test MCTS vs MCTS games
- [ ] Test MCTS vs Random (should win)
- [ ] Test with different iteration counts
- [ ] Test time limits work correctly
- [ ] Run all tests: `cargo test`

### Step 7.3: Performance Benchmarks

**Task:** Measure and optimize performance.

**Create `benches/mcts_bench.rs`:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use othello::*;
use othello::mcts::MCTSPlayer;

fn bench_mcts_easy(c: &mut Criterion) {
    let game = Game::new();
    let player = MCTSPlayer::with_iterations("Bench", 100);
    
    c.bench_function("mcts_100_iterations", |b| {
        b.iter(|| {
            player.choose_move(black_box(&game));
        });
    });
}

criterion_group!(benches, bench_mcts_easy);
criterion_main!(benches);
```

**Action Items:**
- [ ] Add `criterion` to `Cargo.toml` dev-dependencies
- [ ] Create benchmark file
- [ ] Benchmark different iteration counts
- [ ] Benchmark with/without heuristics
- [ ] Identify performance bottlenecks

---

## Phase 8: Performance Optimization

**Estimated Time:** 3-4 hours  
**Goal:** Optimize MCTS for better performance.

### Step 8.1: Optimize Game Cloning

**Issue:** Cloning `Game` for each child node is expensive.

**Solutions:**
1. Cache valid moves in nodes
2. Use immutable data structures (future)
3. Minimize clones where possible

**Action Items:**
- [ ] Profile with `cargo bench` or `perf`
- [ ] Identify hot paths
- [ ] Cache valid moves in `MCTSNode`
- [ ] Reduce unnecessary clones
- [ ] Measure improvement

### Step 8.2: Add Difficulty Presets

**Task:** Create preset difficulty levels.

**Add to `src/mcts/player.rs`:**
```rust
impl MCTSPlayer {
    pub fn easy() -> Self {
        Self::with_iterations("MCTS (Easy)", 200)
            .with_exploration(2.0)
            .with_heuristics(false)
    }
    
    pub fn medium() -> Self {
        Self::with_iterations("MCTS (Medium)", 1000)
            .with_exploration(1.414)
            .with_heuristics(true)
    }
    
    pub fn hard() -> Self {
        Self::with_iterations("MCTS (Hard)", 3000)
            .with_exploration(1.414)
            .with_heuristics(true)
    }
    
    pub fn expert() -> Self {
        Self::with_iterations("MCTS (Expert)", 10000)
            .with_exploration(1.0)
            .with_heuristics(true)
            .with_time_limit_ms(5000)
    }
    
    // Add builder methods...
    pub fn with_exploration(mut self, c: f64) -> Self {
        self.exploration_constant = c;
        self
    }
    
    pub fn with_time_limit_ms(mut self, ms: u64) -> Self {
        self.max_time_ms = Some(ms);
        self
    }
    
    pub fn with_heuristics(mut self, use: bool) -> Self {
        self.use_heuristics = use;
        self
    }
}
```

**Action Items:**
- [ ] Implement difficulty presets
- [ ] Test each difficulty level
- [ ] Tune iteration counts and parameters
- [ ] Document difficulty differences

---

## Phase 9: Polish and Documentation

**Estimated Time:** 2-3 hours  
**Goal:** Final polish and documentation.

### Step 9.1: Add Documentation

**Task:** Add rustdoc comments to all public APIs.

**Example:**
```rust
/// A Monte Carlo Tree Search player for Othello.
///
/// MCTS builds a search tree by repeatedly performing:
/// 1. Selection: Traverse from root to leaf using UCB1
/// 2. Expansion: Add children to leaf node
/// 3. Simulation: Play random game to completion
/// 4. Backpropagation: Update statistics up the tree
///
/// # Example
/// ```
/// use othello::mcts::MCTSPlayer;
/// use othello::Game;
///
/// let game = Game::new();
/// let player = MCTSPlayer::medium();
/// let move_opt = player.choose_move(&game);
/// ```
pub struct MCTSPlayer {
    // ...
}
```

**Action Items:**
- [ ] Add module-level documentation
- [ ] Document all public structs and methods
- [ ] Add code examples
- [ ] Run `cargo doc` to verify

### Step 9.2: Update README

**Task:** Update project README with MCTS information.

**Action Items:**
- [ ] Add MCTS section to README
- [ ] Explain how to use MCTS player
- [ ] Document difficulty levels
- [ ] Add examples

### Step 9.3: Final Testing

**Task:** Comprehensive final test suite.

**Checklist:**
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Benchmarks run successfully
- [ ] No clippy warnings
- [ ] Code compiles with `--release`
- [ ] Manual gameplay test (play a few games)

---

## Implementation Timeline

| Phase | Estimated Time | Cumulative Time |
|-------|---------------|-----------------|
| Phase 1: Module Setup | 2-3 hours | 2-3 hours |
| Phase 2: Core Algorithm | 4-6 hours | 6-9 hours |
| Phase 3: Selection/Expansion | 3-4 hours | 9-13 hours |
| Phase 4: Simulation/Backprop | 4-5 hours | 13-18 hours |
| Phase 5: Player Integration | 3-4 hours | 16-22 hours |
| Phase 6: Heuristics | 5-6 hours | 21-28 hours |
| Phase 7: Testing | 4-5 hours | 25-33 hours |
| Phase 8: Optimization | 3-4 hours | 28-37 hours |
| Phase 9: Polish | 2-3 hours | 30-40 hours |

**Total Estimated Time:** 30-40 hours (approximately 1 week of full-time work)

---

## Common Pitfalls and Solutions

### Issue: Rust Ownership with Tree Traversal

**Problem:** Can't hold mutable references to multiple nodes simultaneously.

**Solution:** Use path-based approach (store indices, then traverse).

### Issue: Simulation Never Ends

**Problem:** Infinite loop in simulation if game state check is wrong.

**Solution:** Always check `GameState::Playing` and handle skip_turn correctly.

### Issue: Result Flipping in Backpropagation

**Problem:** Win/loss gets confused when propagating up tree.

**Solution:** Track which player's perspective each node represents, flip appropriately.

### Issue: Performance Too Slow

**Problem:** Too many clones or iterations.

**Solution:** 
- Cache valid moves
- Reduce iteration count for testing
- Use heuristics to shorten simulations
- Profile to find bottlenecks

---

## Next Steps After Implementation

1. **Test against human players** - Gather feedback on difficulty
2. **Tune parameters** - Adjust iteration counts, exploration constant
3. **Add progress indicators** - Show thinking progress in GUI
4. **Tree reuse** - Keep tree between moves (advanced)
5. **Parallel MCTS** - Use multiple threads (advanced)

---

## Revision History

- **v1.0** (Initial): Complete implementation strategy document created


# Monte Carlo Tree Search (MCTS) Design Document

## 1. Overview

This document outlines the design and implementation strategy for adding a Monte Carlo Tree Search (MCTS) AI opponent to the Othello game. MCTS is a heuristic search algorithm that has proven highly effective in game-playing applications, particularly for games with high branching factors and incomplete information.

### Objectives
- Implement a strong AI opponent using MCTS algorithm
- Integrate seamlessly with the existing `PlayerTrait` interface
- Provide configurable difficulty levels through simulation parameters
- Maintain good performance with reasonable computation time per move
- Support both single-threaded and potential future multi-threaded execution

### Why MCTS for Othello?

MCTS is well-suited for Othello because:
- **High branching factor**: Othello typically has 5-15 valid moves per turn, making exhaustive search impractical
- **Position evaluation difficulty**: Static evaluation functions struggle with Othello's dynamic nature
- **No need for perfect play**: MCTS provides strong play without needing perfect endgame databases
- **Adaptive**: MCTS automatically focuses computation on promising lines of play
- **Proven effectiveness**: MCTS-based engines (e.g., Crazy Stone) have achieved strong performance in Othello

## 2. MCTS Algorithm Overview

Monte Carlo Tree Search builds a search tree through repeated iterations of four phases:

1. **Selection**: Traverse from root to a leaf node using a selection policy (UCB1)
2. **Expansion**: Add one or more child nodes to the leaf
3. **Simulation**: Play out a random game from the new node to a terminal state
4. **Backpropagation**: Update statistics along the path from leaf to root

### Key Components

```
┌─────────────────────────────────────────┐
│         MCTS Tree Structure             │
│  ┌──────────────────────────────────┐   │
│  │        MCTSNode                  │   │
│  │  - visits: usize                 │   │
│  │  - wins: f64                     │   │
│  │  - game_state: Game              │   │
│  │  - children: Vec<MCTSNode>       │   │
│  │  - parent: Option<*MCTSNode>     │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         MCTSPlayer                      │
│  - iterations: usize                    │
│  - exploration_constant: f64            │
│  - max_time_ms: Option<u64>             │
│  - use_heuristics: bool                 │
└─────────────────────────────────────────┘
```

## 3. Architecture Integration

### 3.1 Integration with Existing Architecture

The MCTS implementation will integrate with the existing player system:

```
┌─────────────────────────────────────┐
│      Player Interface Layer         │
│  ┌──────────┐  ┌──────────┐        │
│  │  Human   │  │   AI     │        │
│  └──────────┘  └────┬─────┘        │
│                     │              │
│                     ▼              │
│              ┌──────────────┐      │
│              │  AI Players  │      │
│              ├──────────────┤      │
│              │ RandomAI     │      │
│              │ MCTSPlayer   │      │
│              │ (Future:     │      │
│              │  MinimaxAI)  │      │
│              └──────────────┘      │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         Game Engine (Core)          │
│  ┌──────────┐  ┌──────────┐        │
│  │  Board   │  │  Rules   │        │
│  │  Game    │  │          │        │
│  └──────────┘  └──────────┘        │
└─────────────────────────────────────┘
```

### 3.2 Module Structure

```
src/
├── mcts/
│   ├── mod.rs              # Module declaration
│   ├── node.rs             # MCTSNode implementation
│   ├── player.rs           # MCTSPlayer (implements PlayerTrait)
│   ├── selection.rs        # Selection policies (UCB1, etc.)
│   ├── simulation.rs       # Simulation/rollout strategies
│   └── heuristics.rs       # Position evaluation heuristics
├── ai_player.rs            # Existing random AI (for comparison)
└── player.rs               # PlayerTrait (no changes needed)
```

## 4. Data Structures

### 4.1 MCTSNode

```rust
use crate::game::{Game, Player};
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
```

### 4.2 MCTSPlayer

```rust
use crate::player::PlayerTrait;
use crate::game::Game;
use crate::board::Position;

pub struct MCTSPlayer {
    name: String,
    
    /// Number of MCTS iterations per move
    iterations: usize,
    
    /// Exploration constant for UCB1 formula (typically 1.414 = √2)
    exploration_constant: f64,
    
    /// Maximum time per move in milliseconds (None = no limit)
    max_time_ms: Option<u64>,
    
    /// Whether to use heuristic-based simulation (instead of pure random)
    use_heuristics: bool,
    
    /// Random number generator
    rng: ThreadRng,
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
            use_heuristics: true,
            rng: thread_rng(),
        }
    }
    
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
    
    pub fn set_exploration_constant(&mut self, c: f64) {
        self.exploration_constant = c;
    }
    
    pub fn set_max_time_ms(&mut self, ms: Option<u64>) {
        self.max_time_ms = ms;
    }
    
    pub fn set_use_heuristics(&mut self, use: bool) {
        self.use_heuristics = use;
    }
}
```

## 5. MCTS Algorithm Details

### 5.1 Main MCTS Loop

```rust
impl MCTSPlayer {
    fn mcts_search(&mut self, root_game: &Game) -> Option<Position> {
        let mut root = MCTSNode::new(root_game.clone());
        let start_time = Instant::now();
        
        for iteration in 0..self.iterations {
            // Check time limit
            if let Some(max_ms) = self.max_time_ms {
                if start_time.elapsed().as_millis() as u64 > max_ms {
                    break;
                }
            }
            
            // Selection: Find leaf node
            // Note: This is simplified - actual implementation needs careful ownership handling
            let leaf_node = self.select_leaf(&mut root);
            
            // Expansion: Add children if not terminal
            if !leaf_node.is_terminal && !leaf_node.is_expanded {
                leaf_node.expand();
                leaf_node.is_expanded = true;
            }
            
            // Simulation: Play random game from this node
            let mut sim_game = leaf_node.game_state.clone();
            let result = self.simulate(&mut sim_game, root.current_player);
            
            // Backpropagation: Update statistics along path
            self.backpropagate(&mut root, result);
        }
        
        // Return most visited child (robust child)
        root.best_child_robust().and_then(|n| n.move_from_parent)
    }
}
```

### 5.2 Selection Policy (UCB1)

```rust
impl MCTSNode {
    fn ucb1_value(&self, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY; // Unvisited nodes prioritized
        }
        
        let exploitation = self.value / self.visits as f64;
        let exploration = exploration_constant * 
            ((self.parent_visits().ln()) / self.visits as f64).sqrt();
        
        exploitation + exploration
    }
    
    fn select_child_mut(&mut self, exploration_constant: f64) -> &mut MCTSNode {
        self.children.iter_mut()
            .max_by(|a, b| {
                a.ucb1_value(exploration_constant)
                    .partial_cmp(&b.ucb1_value(exploration_constant))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("select_child called on node with no children")
    }
    
    fn best_child_robust(&self) -> Option<&MCTSNode> {
        // Return child with most visits
        self.children.iter()
            .max_by_key(|c| c.visits)
    }
    
    fn parent_visits(&self) -> usize {
        // Sum of all children's visits
        self.children.iter().map(|c| c.visits).sum::<usize>() + 1
    }
}
```

### 5.3 Expansion

```rust
impl MCTSNode {
    fn expand(&mut self) {
        if self.is_terminal || self.is_expanded {
            return;
        }
        
        let valid_moves = self.game_state.get_valid_moves();
        
        for position in valid_moves {
            let mut child_game = self.game_state.clone();
            // Make the move
            if child_game.make_move(position).is_ok() {
                let child = MCTSNode {
                    visits: 0,
                    value: 0.0,
                    game_state: child_game,
                    move_from_parent: Some(position),
                    children: Vec::new(),
                    is_expanded: false,
                    current_player: self.current_player.opposite(),
                    is_terminal: matches!(child_game.get_game_state(), GameState::GameOver { .. }),
                };
                self.children.push(Box::new(child));
            }
        }
        
        self.is_expanded = true;
    }
}
```

### 5.4 Simulation/Rollout

```rust
impl MCTSPlayer {
    fn simulate(&mut self, game: &mut Game, root_player: Player) -> f64 {
        // Play random game to completion
        while matches!(game.get_game_state(), GameState::Playing) {
            let valid_moves = game.get_valid_moves();
            
            if valid_moves.is_empty() {
                game.skip_turn().ok();
                continue;
            }
            
            // Choose move based on strategy
            let move_pos = if self.use_heuristics {
                self.heuristic_move_selection(game, &valid_moves)
            } else {
                valid_moves.choose(&mut self.rng).copied()
            };
            
            if let Some(pos) = move_pos {
                game.make_move(pos).ok();
            } else {
                game.skip_turn().ok();
            }
        }
        
        // Return result from root player's perspective
        match game.get_game_state() {
            GameState::GameOver { winner } => {
                match winner {
                    Some(player) if player == root_player => 1.0, // Win
                    Some(_) => 0.0, // Loss
                    None => 0.5, // Draw
                }
            }
            _ => 0.5, // Should not happen
        }
    }
}
```

### 5.5 Heuristic Move Selection

For better simulation quality, use heuristics instead of pure random play:

```rust
impl MCTSPlayer {
    fn heuristic_move_selection(&mut self, game: &Game, moves: &[Position]) -> Option<Position> {
        // Prioritize moves based on heuristics
        let scored_moves: Vec<(Position, f64)> = moves.iter()
            .map(|&pos| (pos, self.evaluate_move(game, pos)))
            .collect();
        
        // Weighted random selection based on scores
        // Higher score = higher probability
        let total_score: f64 = scored_moves.iter().map(|(_, score)| score).sum();
        if total_score == 0.0 {
            return moves.choose(&mut self.rng).copied();
        }
        
        let mut r: f64 = self.rng.gen();
        r *= total_score;
        
        let mut cumulative = 0.0;
        for (pos, score) in scored_moves {
            cumulative += score;
            if r <= cumulative {
                return Some(pos);
            }
        }
        
        moves.last().copied()
    }
    
    fn evaluate_move(&self, game: &Game, position: Position) -> f64 {
        // Combine multiple heuristics
        let corner_value = self.corner_heuristic(position);
        let mobility_value = self.mobility_heuristic(game, position);
        let stability_value = self.stability_heuristic(game, position);
        
        corner_value * 10.0 + mobility_value * 2.0 + stability_value * 1.0
    }
}
```

## 6. Heuristics for Othello

### 6.1 Corner Heuristic

Corners are extremely valuable in Othello:

```rust
fn corner_heuristic(&self, position: Position) -> f64 {
    let corners = [
        Position::new(0, 0), Position::new(0, 7),
        Position::new(7, 0), Position::new(7, 7),
    ];
    
    if corners.contains(&position) {
        1.0
    } else if self.is_adjacent_to_corner(position) {
        -0.5  // Adjacent to corner is often bad
    } else {
        0.0
    }
}
```

### 6.2 Mobility Heuristic

More moves = better position:

```rust
fn mobility_heuristic(&self, game: &Game, position: Position) -> f64 {
    let mut test_game = game.clone();
    test_game.make_move(position).ok();
    
    let my_moves = test_game.get_valid_moves().len();
    test_game.skip_turn().ok();
    let opponent_moves = test_game.get_valid_moves().len();
    
    if my_moves + opponent_moves == 0 {
        0.0
    } else {
        (my_moves as f64 - opponent_moves as f64) / (my_moves + opponent_moves) as f64
    }
}
```

### 6.3 Stability Heuristic

Pieces that are harder to flip are more stable:

```rust
fn stability_heuristic(&self, game: &Game, position: Position) -> f64 {
    // Simple implementation: pieces on edges are more stable
    let row = position.row;
    let col = position.col;
    
    let edge_bonus = if row == 0 || row == 7 || col == 0 || col == 7 {
        0.3
    } else {
        0.0
    };
    
    // Pieces near corners on edges are very stable
    let corner_adjacent = (row == 0 && (col == 1 || col == 6)) ||
                          (row == 7 && (col == 1 || col == 6)) ||
                          (col == 0 && (row == 1 || row == 6)) ||
                          (col == 7 && (row == 1 || row == 6));
    
    if corner_adjacent {
        -0.2  // Actually risky (X-square)
    } else {
        edge_bonus
    }
}
```

## 7. Implementation Phases

### Phase 1: Core MCTS Structure (Week 1)
- [ ] Create `mcts/` module directory structure
- [ ] Implement `MCTSNode` with basic structure
- [ ] Implement tree traversal (selection)
- [ ] Implement expansion logic
- [ ] Basic random simulation
- [ ] Backpropagation logic

### Phase 2: UCB1 Selection (Week 1-2)
- [ ] Implement UCB1 formula
- [ ] Implement robust child selection
- [ ] Add unit tests for selection policies
- [ ] Benchmark selection performance

### Phase 3: Integration (Week 2)
- [ ] Implement `MCTSPlayer` struct
- [ ] Implement `PlayerTrait` for `MCTSPlayer`
- [ ] Integrate with existing game loop
- [ ] Add configuration parameters (iterations, time limits)
- [ ] Test against random AI

### Phase 4: Heuristics (Week 2-3)
- [ ] Implement corner heuristic
- [ ] Implement mobility heuristic
- [ ] Implement stability heuristic
- [ ] Implement weighted move selection in simulations
- [ ] Tune heuristic weights

### Phase 5: Performance Optimization (Week 3)
- [ ] Optimize game state cloning (consider immutable updates)
- [ ] Cache valid moves calculation
- [ ] Optimize tree node memory usage
- [ ] Add parallel simulation support (future)
- [ ] Benchmark and profile

### Phase 6: Testing & Tuning (Week 3-4)
- [ ] Unit tests for all MCTS components
- [ ] Integration tests with game engine
- [ ] Performance benchmarks
- [ ] Tune exploration constant
- [ ] Tune iteration counts for difficulty levels
- [ ] Test against known strong positions

### Phase 7: Polish & Documentation (Week 4)
- [ ] Add rustdoc documentation
- [ ] Create difficulty presets (Easy, Medium, Hard, Expert)
- [ ] Add progress indicators for long computations
- [ ] Update README with MCTS information
- [ ] Add examples and usage documentation

## 8. Performance Considerations

### 8.1 Computational Complexity

- **Time per iteration**: ~1-5ms depending on game state
- **Typical iterations**: 500-5000 for strong play
- **Memory per node**: ~200-500 bytes
- **Typical tree size**: 1,000-10,000 nodes

### 8.2 Optimization Strategies

1. **Game State Cloning**: Minimize clones using references where possible
2. **Move Caching**: Cache valid moves in nodes
3. **Early Termination**: Stop early if a move is clearly best
4. **Tree Reuse**: Keep tree between moves (future enhancement)
5. **Parallel Simulations**: Run multiple simulations concurrently (future)

### 8.3 Difficulty Levels

```rust
impl MCTSPlayer {
    pub fn easy() -> Self {
        Self::with_iterations("MCTS (Easy)", 200)
            .with_exploration(2.0)  // More exploration
            .with_heuristics(false) // No heuristics
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
            .with_exploration(1.0)  // Less exploration, more exploitation
            .with_heuristics(true)
            .with_time_limit_ms(5000)  // 5 second limit
    }
    
    fn select_leaf<'a>(&self, node: &'a mut MCTSNode) -> &'a mut MCTSNode {
        // Simplified - actual implementation needs proper tree traversal
        // This is a placeholder showing the concept
        node
    }
    
    fn backpropagate(&self, node: &mut MCTSNode, value: f64) {
        // Simplified - actual implementation needs proper tree traversal
        // This is a placeholder showing the concept
        node.visits += 1;
        node.value += value;
    }
}
```

## 9. Testing Strategy

### 9.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ucb1_selection() {
        // Test UCB1 formula correctness
    }
    
    #[test]
    fn test_node_expansion() {
        // Test that expansion creates correct children
    }
    
    #[test]
    fn test_simulation_completes() {
        // Test that simulation always reaches terminal state
    }
    
    #[test]
    fn test_backpropagation() {
        // Test that statistics update correctly
    }
}
```

### 9.2 Integration Tests

```rust
#[test]
fn test_mcts_vs_random() {
    // MCTS should win significantly more than random
    let mut mcts_wins = 0;
    for _ in 0..100 {
        let mut game = Game::new();
        let mcts = MCTSPlayer::medium();
        let random = AIPlayer::new("Random");
        
        // Play game...
        // Count wins
    }
    
    assert!(mcts_wins > 70); // Should win at least 70% of games
}
```

### 9.3 Performance Benchmarks

```rust
#[bench]
fn bench_mcts_search(b: &mut Bencher) {
    let game = Game::new();
    let mut player = MCTSPlayer::medium();
    
    b.iter(|| {
        player.choose_move(&game);
    });
}
```

## 10. Dependencies

### New Dependencies

```toml
[dependencies]
# Existing dependencies...
rand = "0.8"  # Already included, used for MCTS simulations

# No additional dependencies needed for basic MCTS!
# Future: Consider adding for parallel execution:
# rayon = "1.8"  # For parallel simulations (future enhancement)
```

## 11. Usage Examples

### 11.1 Basic Usage

```rust
use othello::{Game, Player};
use othello::mcts::MCTSPlayer;

fn main() {
    let mut game = Game::new();
    let human = HumanPlayer::new("Player");
    let ai = MCTSPlayer::medium();
    
    // Play game...
}
```

### 11.2 Custom Configuration

```rust
let mut ai = MCTSPlayer::with_iterations("Custom MCTS", 5000);
ai.set_exploration_constant(1.0);
ai.set_max_time_ms(Some(3000)); // 3 second limit
ai.set_use_heuristics(true);
```

### 11.3 Different Difficulty Levels

```rust
let easy_ai = MCTSPlayer::easy();      // ~200 iterations, no heuristics
let medium_ai = MCTSPlayer::medium();  // ~1000 iterations
let hard_ai = MCTSPlayer::hard();      // ~3000 iterations
let expert_ai = MCTSPlayer::expert();  // ~10000 iterations, 5s limit
```

## 12. Future Enhancements

### 12.1 Advanced MCTS Variants

1. **UCT (Upper Confidence bounds applied to Trees)**: Standard variant (already planned)
2. **Progressive Bias**: Use heuristics to bias initial node values
3. **RAVE (Rapid Action Value Estimation)**: Share statistics across similar moves
4. **Progressive Unpruning**: Gradually expand tree depth

### 12.2 Parallelization

1. **Parallel Simulations**: Run multiple rollouts concurrently
2. **Tree Parallelism**: Multiple threads building different parts of tree
3. **Virtual Loss**: Prevent threads from selecting same nodes

### 12.3 Tree Reuse

1. **Keep tree between moves**: Reuse subtree for new root
2. **Gradual tree growth**: Maintain tree across entire game
3. **Tree pruning**: Remove nodes that are no longer relevant

### 12.4 Learning

1. **Self-play training**: Improve heuristics through self-play
2. **Opening book**: Pre-computed strong opening moves
3. **Endgame database**: Perfect play in endgame positions

## 13. References

1. Browne, C., et al. (2012). "A Survey of Monte Carlo Tree Search Methods"
2. Chaslot, G., et al. (2008). "Monte-Carlo Tree Search: A New Framework for Game AI"
3. Kocsis, L., & Szepesvári, C. (2006). "Bandit based Monte-Carlo Planning"
4. Othello strategy guides and heuristics

## 14. Appendix: UCB1 Formula

The UCB1 (Upper Confidence Bound 1) formula balances exploitation and exploration:

```
UCB1 = X̄ᵢ + C × √(ln(N) / nᵢ)

Where:
- X̄ᵢ = average value of node i (wins/visits)
- C = exploration constant (typically √2 ≈ 1.414)
- N = total visits to parent node
- nᵢ = visits to node i
```

This formula ensures that:
- Nodes with high win rates are exploited
- Unvisited nodes are explored
- Nodes with few visits get more exploration

## Revision History

- **v1.0** (Initial): Complete MCTS design document created


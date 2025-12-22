# Othello Game Design Document

## 1. Overview

This document outlines the design and architecture for an Othello (Reversi) game implementation in Rust. The game will be a console-based application that allows two players to compete, with the potential for future enhancements including AI opponents and a graphical interface.

### Objectives
- Implement a complete, playable Othello game
- Demonstrate Rust's strengths in memory safety, performance, and pattern matching
- Create a clean, modular architecture that is easily extensible
- Support both human vs human and human vs AI gameplay

## 2. Game Rules Summary

Othello is played on an 8x8 board with 64 discs. Key rules:
- Players alternate turns (Black starts)
- A valid move must flank one or more opponent pieces between the new piece and another of your color
- All flanked pieces are flipped to the current player's color
- Game ends when neither player can make a valid move
- Player with more pieces of their color wins

## 3. Architecture Overview

The project will follow a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────┐
│         Game Engine (Core)          │
│  ┌──────────┐  ┌──────────┐         │
│  │  Board   │  │  Rules   │         │
│  └──────────┘  └──────────┘         │
│  ┌──────────┐  ┌──────────┐         │
│  │  Move    │  │  Game    │         │
│  └──────────┘  └──────────┘         │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      Player Interface Layer         │
│  ┌──────────┐  ┌──────────┐         │
│  │  Human   │  │   AI     │         │
│  └──────────┘  └──────────┘         │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│       Presentation Layer            │
│  ┌──────────┐  ┌──────────┐         │
│  │ Console  │  │  (GUI)   │         │
│  └──────────┘  └──────────┘         │
└─────────────────────────────────────┘
```

## 4. Core Modules

### 4.1 Board Module (`board.rs`)

**Purpose**: Manages the game board state and basic board operations.

**Key Components**:
- `Board`: Represents the 8x8 game board
- `Position`: A coordinate on the board (row, col)
- `Cell`: Enum representing empty, black, or white

**Key Functions**:
- `new()`: Create empty board with initial setup
- `get_cell(position)`: Get cell state at position
- `set_cell(position, cell)`: Set cell state
- `is_valid_position(position)`: Check bounds
- `count_pieces(color)`: Count pieces of a given color
- `clone()`: Create a copy of the board

**Data Structure**:
```rust
pub struct Board {
    grid: [[Cell; 8]; 8],
}

pub enum Cell {
    Empty,
    Black,
    White,
}

pub struct Position {
    pub row: usize,
    pub col: usize,
}
```

### 4.2 Rules Module (`rules.rs`)

**Purpose**: Implements game logic and rule validation.

**Key Functions**:
- `is_valid_move(board, position, player)`: Validate if move is legal
- `get_valid_moves(board, player)`: Find all valid moves for a player
- `apply_move(board, position, player)`: Execute move and flip pieces
- `get_flipped_positions(board, position, player)`: Calculate which pieces flip
- `is_game_over(board)`: Check if game has ended
- `get_winner(board)`: Determine winner (if game is over)
- `has_valid_move(board, player)`: Check if player can move

**Algorithm for Move Validation**:
1. Check if target cell is empty
2. Check 8 directions (N, NE, E, SE, S, SW, W, NW)
3. For each direction, traverse until finding:
   - Own color piece (valid flank)
   - Empty cell (invalid)
   - Board edge (invalid)
4. If at least one valid flank exists, move is legal

### 4.3 Game Module (`game.rs`)

**Purpose**: Manages overall game state and flow.

**Key Components**:
- `Game`: Main game state structure
- `GameState`: Enum for game status (Playing, GameOver)
- `Player`: Enum for current player

**Key Functions**:
- `new()`: Initialize new game
- `current_player()`: Get active player
- `make_move(position)`: Attempt to make a move
- `skip_turn()`: Skip when no valid moves
- `get_game_state()`: Get current game state
- `get_board()`: Access board (read-only)
- `get_score()`: Get current score (black, white)

**Data Structure**:
```rust
pub struct Game {
    board: Board,
    current_player: Player,
    game_state: GameState,
}

pub enum Player {
    Black,
    White,
}

pub enum GameState {
    Playing,
    GameOver { winner: Option<Player> },
}
```

### 4.4 Move Module (`move.rs`)

**Purpose**: Represents and validates moves.

**Key Components**:
- `Move`: Represents a move attempt
- `MoveResult`: Result of move attempt (Success, Invalid)

**Key Functions**:
- `from_input(input)`: Parse move from user input (e.g., "a1", "3,4")
- `to_position()`: Convert to board position

## 5. Player Interface

### 5.1 Player Trait (`player.rs`)

**Purpose**: Abstract interface for different player types.

```rust
pub trait Player {
    fn choose_move(&self, game: &Game) -> Option<Position>;
    fn get_name(&self) -> &str;
}
```

### 5.2 Human Player (`human_player.rs`)

**Purpose**: Handles human input.

**Key Functions**:
- `new(name)`: Create human player
- `choose_move(game)`: Prompt for and parse user input
- `prompt_move()`: Display prompt and read input
- `parse_input(input)`: Convert string to Position

### 5.3 AI Player (`ai_player.rs`)

**Purpose**: Implements computer opponent (future enhancement).

**Initial Implementation**: Random valid move

**Future Enhancements**:
- Minimax algorithm
- Alpha-beta pruning
- Position evaluation heuristics
- Difficulty levels

## 6. User Interface

### 6.1 Display Module (`display.rs`)

**Purpose**: Handles rendering the game state to console.

**Key Functions**:
- `render_board(board)`: Display board with coordinates
- `render_score(game)`: Show current scores
- `render_turn(game)`: Show current player
- `render_valid_moves(moves)`: Show available moves
- `render_game_over(game)`: Display winner and final score
- `clear_screen()`: Clear terminal
- `render_prompt()`: Display input prompt

**Display Format**:
```
    A B C D E F G H
  1 . . . . . . . .
  2 . . . . . . . .
  3 . . . W B . . .
  4 . . . B W . . .
  5 . . . . . . . .
  6 . . . . . . . .
  7 . . . . . . . .
  8 . . . . . . . .

  Black: 2  White: 2
  Current Player: Black
  Valid moves: A4, C4, E4, F4, D2, D6, F6, C5, F5
```

## 7. Main Game Loop

### 7.1 Main Module (`main.rs`)

**Purpose**: Entry point and game orchestration.

**Flow**:
1. Initialize game
2. Create players (human/AI)
3. Game loop:
   - Display board and game state
   - Get move from current player
   - Validate and apply move
   - Switch players (if no valid moves, skip turn)
   - Check for game over
4. Display final results

**Pseudocode**:
```rust
fn main() {
    let mut game = Game::new();
    let black_player = HumanPlayer::new("Player 1");
    let white_player = HumanPlayer::new("Player 2");
    
    loop {
        display::render_game(&game);
        
        let player = match game.current_player() {
            Player::Black => &black_player,
            Player::White => &white_player,
        };
        
        if let Some(position) = player.choose_move(&game) {
            game.make_move(position)?;
        } else {
            // Player skipped (no valid moves)
            game.skip_turn();
        }
        
        if let GameState::GameOver { winner } = game.get_game_state() {
            display::render_game_over(&game, winner);
            break;
        }
    }
}
```

## 8. Data Structures Summary

### Core Types

```rust
// Board representation
pub struct Board {
    grid: [[Cell; 8]; 8],
}

pub enum Cell {
    Empty,
    Black,
    White,
}

// Position on board
pub struct Position {
    pub row: usize,  // 0-7
    pub col: usize,  // 0-7
}

// Player
pub enum Player {
    Black,
    White,
}

// Game state
pub enum GameState {
    Playing,
    GameOver { winner: Option<Player> },
}

// Direction for move validation
pub enum Direction {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}
```

## 9. Algorithms

### 9.1 Move Validation Algorithm

```
function is_valid_move(board, position, player):
    if board.get_cell(position) != Empty:
        return false
    
    directions = [N, NE, E, SE, S, SW, W, NW]
    
    for each direction in directions:
        if check_direction(board, position, direction, player):
            return true
    
    return false

function check_direction(board, position, direction, player):
    opponent = opposite(player)
    current = move(position, direction)
    found_opponent = false
    
    while board.is_valid_position(current):
        cell = board.get_cell(current)
        if cell == Empty:
            return false
        if cell == opponent:
            found_opponent = true
            current = move(current, direction)
        else if cell == player:
            return found_opponent
        else:
            return false
    
    return false
```

### 9.2 Flip Pieces Algorithm

```
function apply_move(board, position, player):
    board.set_cell(position, player)
    
    for each direction in directions:
        flipped = get_flipped_in_direction(board, position, direction, player)
        for pos in flipped:
            board.set_cell(pos, player)

function get_flipped_in_direction(board, position, direction, player):
    opponent = opposite(player)
    result = []
    current = move(position, direction)
    
    while board.is_valid_position(current):
        cell = board.get_cell(current)
        if cell == opponent:
            result.push(current)
            current = move(current, direction)
        else if cell == player:
            return result
        else:
            return []
    
    return []
```

## 10. Error Handling

Use Rust's `Result` type for error handling:

```rust
pub enum GameError {
    InvalidMove,
    OutOfBounds,
    GameOver,
    NoValidMoves,
}
```

Key operations return `Result<T, GameError>`:
- `make_move()`: Returns `Result<(), GameError>`
- `apply_move()`: Returns `Result<Vec<Position>, GameError>`

## 11. Testing Strategy

### Unit Tests
- Board operations (set/get, bounds checking)
- Move validation logic
- Piece flipping logic
- Score counting
- Game state transitions

### Integration Tests
- Complete game flow
- Win conditions
- Draw scenarios
- Turn skipping logic

### Test Files Structure
- `board_test.rs`
- `rules_test.rs`
- `game_test.rs`

## 12. Project Structure

```
othello/
├── Cargo.toml
├── README.md
├── DESIGN.md
└── src/
    ├── main.rs
    ├── board.rs
    ├── rules.rs
    ├── game.rs
    ├── move.rs
    ├── player.rs
    ├── human_player.rs
    ├── ai_player.rs
    ├── display.rs
    └── error.rs
```

## 13. Implementation Phases

### Phase 1: Core Engine
- [ ] Board module with basic operations
- [ ] Cell and Position types
- [ ] Basic board display

### Phase 2: Game Rules
- [ ] Move validation algorithm
- [ ] Piece flipping logic
- [ ] Valid moves discovery
- [ ] Game over detection

### Phase 3: Game State Management
- [ ] Game struct and state management
- [ ] Turn management
- [ ] Score tracking

### Phase 4: Human Interface
- [ ] Display module
- [ ] Human player input parsing
- [ ] Basic game loop

### Phase 5: Polish
- [ ] Error handling
- [ ] Input validation
- [ ] Better UI/UX
- [ ] Documentation

### Phase 6: AI (Future)
- [ ] Random AI player
- [ ] Minimax algorithm
- [ ] Difficulty levels

## 14. Rust-Specific Considerations

### Performance
- Use `[[Cell; 8]; 8]` for stack-allocated board (fast, no heap allocation)
- Leverage Rust's zero-cost abstractions
- Use pattern matching for cell state checks

### Memory Safety
- Rust's ownership system prevents common bugs
- Use references for read-only access
- Clone only when necessary (move validation)

### Code Organization
- Use modules for logical separation
- Leverage Rust's type system for safety (enums for state)
- Use `Result` types for error handling

### Idiomatic Rust
- Use pattern matching extensively
- Implement `Display` trait for custom types
- Use `Option` and `Result` appropriately
- Follow Rust naming conventions

## 15. Future Enhancements

1. **AI Improvements**
   - Minimax with alpha-beta pruning
   - Monte Carlo Tree Search (MCTS)
   - Neural network-based evaluation
   - Multiple difficulty levels

2. **UI Improvements**
   - Graphical interface using crates `ggez` and `keyframe`
   - Animation for piece flips (using crate `keyframe`)
   - Move history/replay

3. **Features**
   - Undo/redo functionality
   - Save/load game state
   - Tournament mode
   - Statistics tracking
   - Network multiplayer

4. **Code Quality**
   - Benchmarking for performance-critical paths
   - More comprehensive test coverage
   - Documentation (rustdoc)
   - Clippy linting configuration

## 16. Dependencies

### Minimal Dependencies (Phase 1-4)
- Standard library only (no external dependencies needed)

### Potential Future Dependencies
- `ratatui` or `crossterm`: Advanced terminal UI
- `serde`: Serialization for save/load
- `clap`: Command-line argument parsing
- `rand`: For AI player (random moves initially)

## 17. Example Usage

```rust
use othello::{Game, Player, HumanPlayer};

fn main() {
    let mut game = Game::new();
    let black = HumanPlayer::new("Alice");
    let white = HumanPlayer::new("Bob");
    
    // Game loop handled by Game
    game.play(black, white);
}
```

---

## Revision History

- **v1.0** (Initial): Complete design document created



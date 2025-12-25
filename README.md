# Othello Game in Rust: A journey with Cursor

I was able to implement this in less than a day in a language (Rust) that I have beginner
level knowledge of.  The intent of this is to demonstrate what can be done with cursor
in a few hours.

An implementation of the classic Othello (Reversi) game written in Rust with a graphical interface.

## Features

- Full Othello game implementation with proper rules
- **Two interface options:**
  - **Graphical version (Default)** - Full GUI with animations (requires Rust 1.81+)
  - **Console version** - Enhanced terminal UI with colors and Unicode symbols
- Enhanced console-based user interface with:
  - **Colorized terminal output** - Beautiful colors for pieces, scores, and indicators
  - **Visual piece symbols** - Uses ○ (white) and ● (black) circles instead of letters
  - **Valid move highlighting** - Green dots (•) show where you can play
  - **Enhanced board borders** - Unicode box-drawing characters for a polished look
  - **Color-coded scores** - Easy-to-read score display with colored symbols
- Graphical interface features:
  - **Smooth animations** - Piece flips animated using keyframe easing functions
  - **Mouse input** - Click to place pieces
  - **Visual feedback** - Valid moves highlighted on the board
  - **Modern UI** - Clean, colorful interface with score display
- Support for human vs human gameplay
- **Monte Carlo Tree Search (MCTS) AI player** - Advanced AI with multiple difficulty levels
- Modular, extensible architecture
- Error handling with Rust's Result types

## Installation

Make sure you have Rust installed. If not, install it from [rustup.rs](https://rustup.rs/).

## Building

```bash
cargo build --release
```

## Running

### Graphical Version (Default)

The GUI version is now the default interface. Simply run:

```bash
cargo run --bin othello
```

Or run the release version:

```bash
cargo run --bin othello --release
```

**Note:** The GUI version requires Rust 1.81 or later. If you have an older Rust version, use the console version instead.

### Console Version

To use the console-based interface instead:

```bash
cargo run --bin othello-console
```

Or run the release version:

```bash
cargo run --bin othello-console --release
```

The graphical version provides:
- Beautiful visual interface with colored board
- Smooth piece flip animations using keyframe easing
- Mouse-based move selection
- Visual indicators for valid moves
- Score and game state display

## How to Play

1. The game starts with Black player's turn
2. Enter your move using one of these formats:
   - Letter-number format: `d3`, `a1`, `h8` (column letter + row number)
   - Coordinate format: `3,3` or `3 3` (row, col, both 0-7 or 1-8)
3. Valid moves are shown after each turn
4. If you have no valid moves, your turn will be skipped automatically
5. The game ends when neither player can make a move
6. The player with more pieces wins

## Example Moves

- `d3` - Place piece at column D, row 3
- `3,3` - Place piece at row 3, column 3 (0-indexed)
- `q` or `quit` - Quit the game

## Game Rules

- Players alternate turns (Black starts)
- A valid move must flank one or more opponent pieces
- All flanked pieces are flipped to your color
- Game ends when neither player can make a valid move
- Player with more pieces wins

## MCTS AI Player

The game includes a sophisticated Monte Carlo Tree Search (MCTS) AI player with multiple difficulty levels.

### Quick Start

```rust
use othello::{Game, PlayerTrait};
use othello::MCTSPlayer;

let game = Game::new();
let player = MCTSPlayer::medium();  // Medium difficulty
let move_opt = player.choose_move(&game);
```

### Difficulty Levels

The MCTS player provides four preset difficulty levels:

- **Easy** (`MCTSPlayer::easy()`) - 200 iterations, ~200-500ms per move
  - Higher exploration for more varied play
  - No heuristics (pure random simulation)
  - Good for quick games or beginners

- **Medium** (`MCTSPlayer::medium()`) - 1000 iterations, ~1-3s per move
  - Balanced settings with heuristics enabled
  - Recommended for most games

- **Hard** (`MCTSPlayer::hard()`) - 3000 iterations, ~3-10s per move
  - Strong play suitable for experienced players
  - Heuristics enabled for better simulation quality

- **Expert** (`MCTSPlayer::expert()`) - 10000 iterations, ~10-30s per move (capped at 5s)
  - Very strong play for expert-level competition
  - Lower exploration for focused exploitation
  - Time limit prevents excessively long moves

### Custom Configuration

You can also create custom MCTS players with fine-tuned parameters:

```rust
use othello::MCTSPlayer;

let player = MCTSPlayer::with_iterations("Custom AI", 2000)
    .with_exploration(1.5)      // Exploration constant (default: √2 ≈ 1.414)
    .with_time_limit_ms(3000)   // Maximum time per move
    .with_heuristics(true);     // Enable heuristic-guided simulation
```

### How MCTS Works

MCTS builds a search tree by repeatedly performing four phases:

1. **Selection**: Traverse from root to leaf using UCB1 (Upper Confidence Bound)
2. **Expansion**: Add children to the selected leaf node
3. **Simulation**: Play a random/heuristic-guided game to completion
4. **Backpropagation**: Update statistics (visits and win rate) up the tree

After many iterations, the move from the most-visited child is selected. The algorithm balances exploration (trying new moves) with exploitation (choosing promising moves).

### Heuristics

When enabled, the MCTS player uses Othello-specific heuristics to improve simulation quality:

- **Corner heuristic**: Prioritizes corner positions (very valuable) and avoids adjacent squares (risky)
- **Mobility heuristic**: Prefers moves that maximize future move options
- **Stability heuristic**: Favors edge pieces which are harder to flip

### Performance Tips

- **Iterations**: More iterations generally lead to better moves but take longer (diminishing returns after ~5000)
- **Time limits**: Use `with_time_limit_ms()` to cap move time for responsive gameplay
- **Heuristics**: Enable heuristics to improve simulation quality and move selection
- **Exploration constant**: Lower values (1.0-1.2) favor exploitation, higher values (2.0+) favor exploration

For library usage, see the [`mcts` module documentation](https://docs.rs/othello/latest/othello/mcts/index.html).

## Project Structure

```
src/
├── main.rs          # Entry point and game loop
├── lib.rs           # Library root with module declarations
├── board.rs         # Board representation and operations
├── rules.rs         # Game rules and move validation
├── game.rs          # Game state management
├── player.rs        # Player trait definition
├── human_player.rs  # Human player implementation
├── ai_player.rs     # AI player implementation (random)
├── display.rs       # Console rendering
├── error.rs         # Error types
└── mcts/            # Monte Carlo Tree Search AI
    ├── mod.rs       # Module declarations
    ├── node.rs      # MCTS tree node implementation
    ├── player.rs    # MCTS player implementation
    └── heuristics.rs # Move evaluation heuristics
```

## UI Features

The game features an enhanced terminal interface:

- **Colors**: Automatic color detection (disabled if `NO_COLOR` environment variable is set)
- **Piece Display**: 
  - `●` Black pieces (bright black/gray)
  - `○` White pieces (bright white)
  - `•` Valid moves (bright green)
  - `·` Empty spaces (dim gray)
- **Board Layout**: Clean borders with Unicode box-drawing characters
- **Information Display**: Color-coded scores, current player, and valid moves

## Future Enhancements

- Alpha-beta pruning for alternative AI approach
- Tree reuse between moves (advanced MCTS optimization)
- Parallel MCTS (multi-threaded search)
- Move history/replay
- Save/load game state
- Undo/redo functionality
- Network multiplayer

## License

This project is open source and available for educational purposes.


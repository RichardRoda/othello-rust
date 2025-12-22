# Othello Game in Rust

A console-based implementation of the classic Othello (Reversi) game written in Rust.

## Features

- Full Othello game implementation with proper rules
- Console-based user interface
- Support for human vs human gameplay
- Modular, extensible architecture
- Error handling with Rust's Result types

## Installation

Make sure you have Rust installed. If not, install it from [rustup.rs](https://rustup.rs/).

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run
```

Or run the release version:

```bash
cargo run --release
```

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
└── error.rs         # Error types
```

## Future Enhancements

- AI with minimax algorithm
- Alpha-beta pruning for better AI performance
- Graphical interface
- Save/load game state
- Undo/redo functionality
- Network multiplayer

## License

This project is open source and available for educational purposes.


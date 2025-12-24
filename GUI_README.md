# Othello GUI Version

## Overview

This directory contains the graphical user interface implementation for the Othello game, using `ggez` for rendering and `keyframe` for animations.

## Requirements

- **Rust 1.81 or later** - Required for ggez 0.8 dependencies
- To check your Rust version: `rustc --version`
- To update Rust: `rustup update stable`

## Building the GUI Version

```bash
# Build with GUI features enabled
cargo build --bin othello-gui --features gui --release

# Run the GUI version
cargo run --bin othello-gui --features gui --release
```

## Features

### Visual Interface
- **Green game board** - Classic Othello board appearance
- **Colored pieces** - Black (●) and White (○) pieces with borders
- **Valid move indicators** - Green dots show where you can play
- **Score display** - Real-time score tracking
- **Game state display** - Shows current player and game over messages

### Animations
- **Piece flip animations** - Smooth scaling animations when pieces are flipped
- **Keyframe easing** - Uses `EaseInOutCubic` for smooth, natural motion
- **300ms duration** - Quick but visible animations

### Controls
- **Mouse click** - Click on a valid move position to place a piece
- **Automatic turn management** - Game handles player switching automatically
- **Input blocking** - Clicks are ignored during animations

## Implementation Details

### Graphics Module (`src/graphics.rs`)

The `GraphicsState` struct manages:
- Board rendering with grid lines
- Piece rendering (static and animated)
- Valid move highlighting
- UI text rendering (scores, current player, game over)
- Animation state management

### Animation System

Uses `keyframe` crate's `ease` function with `EaseInOutCubic`:
- Pieces scale from 1.0 → 0.1 → 1.0 during flip
- Smooth transition between colors (black ↔ white)
- Animations complete in 300ms

### Game Loop (`src/main_gui.rs`)

The `MainState` struct implements ggez's `EventHandler` trait:
- `update()` - Updates animation state and game logic
- `draw()` - Renders all graphics
- `mouse_button_down_event()` - Handles move input

## File Structure

```
src/
├── graphics.rs    # Graphics rendering and animation logic
├── main_gui.rs    # GUI game loop and event handling
└── ...            # Other game modules (board, rules, game, etc.)
```

## Troubleshooting

### "rustc 1.80.0 is not supported" Error

This means you need to upgrade Rust:
```bash
rustup update stable
```

### GUI window doesn't appear

- Ensure you're running with `--features gui` flag
- Check that ggez dependencies compiled successfully
- Try running in release mode: `--release`

### Animations not working

- Check that `keyframe` dependency is available
- Verify game logic is calling `add_flip_animation()` correctly
- Ensure animations aren't being cleared prematurely

## Future Enhancements

- [ ] Add sound effects for moves and flips
- [ ] Add animation speed settings
- [ ] Add different animation styles
- [ ] Add board theme options
- [ ] Add piece texture/shadows for 3D effect




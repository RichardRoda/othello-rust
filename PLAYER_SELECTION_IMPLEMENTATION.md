# Player Selection UI Implementation Plan

## Overview

This document outlines the implementation plan for adding player selection functionality to both the GUI and console versions of the Othello game. This will allow users to choose between Human, AI (Random), and MCTS players with different difficulty levels for both Black and White players.

## Current State

### Existing Player Types
- **HumanPlayer** (`src/human_player.rs`) - Interactive human player
- **AIPlayer** (`src/ai_player.rs`) - Random move AI
- **MCTSPlayer** (`src/mcts/player.rs`) - Monte Carlo Tree Search AI with difficulty presets:
  - `easy()` - 200 iterations
  - `medium()` - 1000 iterations
  - `hard()` - 3000 iterations
  - `expert()` - 10000 iterations, 5s time limit

### Current Limitations
- **GUI version** (`src/main_gui.rs`): Hardcoded to human vs human
- **Console version** (`src/main.rs`): Hardcoded to human vs human
- No way to select player types before starting a game

## Goals

1. Add player selection UI to both GUI and console versions
2. Allow selection of player type for both Black and White
3. Support all existing player types (Human, AI, MCTS with difficulties)
4. Maintain clean separation between UI and game logic
5. Provide intuitive user experience

---

## Phase 1: Design Player Selection Data Structures

**Estimated Time:** 1-2 hours  
**Goal:** Create types and enums to represent player selection choices.

### Step 1.1: Create Player Selection Types

**File: `src/player_selection.rs` (new file)**

```rust
use crate::player::PlayerTrait;
use crate::{HumanPlayer, AIPlayer};
use crate::mcts::MCTSPlayer;

/// Represents the type of player that can be selected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    AIRandom,
    MCTSEasy,
    MCTSMedium,
    MCTSHard,
    MCTSExpert,
}

impl PlayerType {
    /// Get all available player types
    pub fn all() -> Vec<PlayerType> {
        vec![
            PlayerType::Human,
            PlayerType::AIRandom,
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
        }
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
```

**Action Items:**
- [ ] Create `src/player_selection.rs`
- [ ] Implement `PlayerType` enum with all variants
- [ ] Implement `PlayerConfig` struct
- [ ] Add `pub mod player_selection;` to `src/lib.rs`
- [ ] Export types: `pub use player_selection::{PlayerType, PlayerConfig};`
- [ ] Run `cargo build` to verify compilation

---

## Phase 2: Console Player Selection UI

**Estimated Time:** 2-3 hours  
**Goal:** Add interactive player selection menu to console version.

### Step 2.1: Create Console Selection Module

**File: `src/console_selection.rs` (new file)**

```rust
use crate::player_selection::{PlayerType, PlayerConfig};
use std::io::{self, Write};

/// Display player selection menu and get user choice
pub fn select_player_type(prompt: &str) -> PlayerType {
    let types = PlayerType::all();
    
    loop {
        println!("\n{}", prompt);
        println!("Available player types:");
        
        for (i, player_type) in types.iter().enumerate() {
            println!("  {}. {}", i + 1, player_type.display_name());
        }
        
        print!("Enter choice (1-{}): ", types.len());
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            continue;
        }
        
        if let Ok(choice) = input.trim().parse::<usize>() {
            if choice >= 1 && choice <= types.len() {
                return types[choice - 1];
            }
        }
        
        println!("Invalid choice. Please try again.");
    }
}

/// Get player configuration from user
pub fn get_player_config() -> PlayerConfig {
    println!("\n╔════════════════════════════════════╗");
    println!("║     Player Selection                ║");
    println!("╚════════════════════════════════════╝");
    
    let black_type = select_player_type("Select player type for Black:");
    let white_type = select_player_type("Select player type for White:");
    
    PlayerConfig::new(black_type, white_type)
}
```

**Action Items:**
- [ ] Create `src/console_selection.rs`
- [ ] Implement `select_player_type()` function
- [ ] Implement `get_player_config()` function
- [ ] Add `pub mod console_selection;` to `src/lib.rs`
- [ ] Test selection with various inputs
- [ ] Handle invalid input gracefully

### Step 2.2: Integrate Selection into Console Main

**File: `src/main.rs` (modify)**

**Changes needed:**
1. Import player selection modules
2. Call `get_player_config()` before starting game
3. Use `PlayerConfig::create_players()` instead of hardcoded players

```rust
use othello::{Game, PlayerTrait, console_selection};

fn main() {
    othello::display::clear_screen();
    println!("╔════════════════════════════╗");
    println!("║     Welcome to Othello!    ║");
    println!("╚════════════════════════════╝");
    println!();
    println!("Press Enter to start...");
    let _ = std::io::stdin().read_line(&mut String::new());
    
    // Get player configuration
    let config = console_selection::get_player_config();
    let (black_player, white_player) = config.create_players();
    
    let mut game = Game::new();
    
    // Main game loop
    loop {
        // ... rest of game loop using black_player and white_player
    }
}
```

**Action Items:**
- [ ] Update `src/main.rs` imports
- [ ] Add player selection call before game loop
- [ ] Replace hardcoded players with config-based players
- [ ] Test with different player combinations
- [ ] Verify AI players make moves automatically

---

## Phase 3: GUI Player Selection UI

**Estimated Time:** 4-6 hours  
**Goal:** Add graphical player selection screen to GUI version.

### Step 3.1: Create Selection Screen State

**File: `src/gui_selection.rs` (new file)**

```rust
use ggez::graphics::{Canvas, Color, DrawParam, Text, TextFragment};
use ggez::{Context, GameResult};
use ggez::input::keyboard::KeyInput;
use crate::player_selection::{PlayerType, PlayerConfig};

/// UI state for player selection screen
pub struct SelectionScreen {
    black_selection: usize,
    white_selection: usize,
    player_types: Vec<PlayerType>,
    selected_side: SelectionSide, // Which side is currently being selected
    config: Option<PlayerConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SelectionSide {
    Black,
    White,
    Confirmed,
}

impl SelectionScreen {
    pub fn new() -> Self {
        Self {
            black_selection: 0,
            white_selection: 0,
            player_types: PlayerType::all(),
            selected_side: SelectionSide::Black,
            config: None,
        }
    }
    
    /// Check if selection is complete
    pub fn is_complete(&self) -> bool {
        self.config.is_some()
    }
    
    /// Get the player configuration (if complete)
    pub fn get_config(&self) -> Option<PlayerConfig> {
        self.config.clone()
    }
    
    /// Handle keyboard input
    pub fn handle_key(&mut self, key: KeyInput) {
        match key.keycode {
            Some(ggez::input::keyboard::KeyCode::Up) => {
                self.decrement_selection();
            }
            Some(ggez::input::keyboard::KeyCode::Down) => {
                self.increment_selection();
            }
            Some(ggez::input::keyboard::KeyCode::Return) => {
                self.confirm_selection();
            }
            Some(ggez::input::keyboard::KeyCode::Tab) => {
                self.switch_side();
            }
            _ => {}
        }
    }
    
    fn increment_selection(&mut self) {
        match self.selected_side {
            SelectionSide::Black => {
                self.black_selection = (self.black_selection + 1) % self.player_types.len();
            }
            SelectionSide::White => {
                self.white_selection = (self.white_selection + 1) % self.player_types.len();
            }
            SelectionSide::Confirmed => {}
        }
    }
    
    fn decrement_selection(&mut self) {
        match self.selected_side {
            SelectionSide::Black => {
                if self.black_selection == 0 {
                    self.black_selection = self.player_types.len() - 1;
                } else {
                    self.black_selection -= 1;
                }
            }
            SelectionSide::White => {
                if self.white_selection == 0 {
                    self.white_selection = self.player_types.len() - 1;
                } else {
                    self.white_selection -= 1;
                }
            }
            SelectionSide::Confirmed => {}
        }
    }
    
    fn confirm_selection(&mut self) {
        match self.selected_side {
            SelectionSide::Black => {
                self.selected_side = SelectionSide::White;
            }
            SelectionSide::White => {
                // Create config and mark as complete
                self.config = Some(PlayerConfig::new(
                    self.player_types[self.black_selection],
                    self.player_types[self.white_selection],
                ));
                self.selected_side = SelectionSide::Confirmed;
            }
            SelectionSide::Confirmed => {}
        }
    }
    
    fn switch_side(&mut self) {
        match self.selected_side {
            SelectionSide::Black => self.selected_side = SelectionSide::White,
            SelectionSide::White => self.selected_side = SelectionSide::Black,
            SelectionSide::Confirmed => {}
        }
    }
    
    /// Render the selection screen
    pub fn draw(&self, canvas: &mut Canvas, ctx: &Context) -> GameResult {
        use ggez::graphics::Text;
        
        // Title
        let title = Text::new(TextFragment::new("Select Players")
            .color(Color::WHITE)
            .scale(48.0));
        canvas.draw(&title, DrawParam::default().dest([400.0, 50.0]));
        
        // Black player selection
        let black_label = Text::new(TextFragment::new("Black Player:")
            .color(if self.selected_side == SelectionSide::Black {
                Color::YELLOW
            } else {
                Color::WHITE
            })
            .scale(32.0));
        canvas.draw(&black_label, DrawParam::default().dest([100.0, 200.0]));
        
        // White player selection
        let white_label = Text::new(TextFragment::new("White Player:")
            .color(if self.selected_side == SelectionSide::White {
                Color::YELLOW
            } else {
                Color::WHITE
            })
            .scale(32.0));
        canvas.draw(&white_label, DrawParam::default().dest([100.0, 400.0]));
        
        // Player type options
        for (i, player_type) in self.player_types.iter().enumerate() {
            let y_offset = 250.0 + (i as f32 * 40.0);
            
            // Black selection
            let is_selected_black = self.selected_side == SelectionSide::Black 
                && self.black_selection == i;
            let black_color = if is_selected_black {
                Color::YELLOW
            } else if self.black_selection == i {
                Color::GREEN
            } else {
                Color::GRAY
            };
            
            let black_text = Text::new(TextFragment::new(format!("  {}", player_type.display_name()))
                .color(black_color)
                .scale(24.0));
            canvas.draw(&black_text, DrawParam::default().dest([150.0, y_offset]));
            
            // White selection
            let is_selected_white = self.selected_side == SelectionSide::White 
                && self.white_selection == i;
            let white_color = if is_selected_white {
                Color::YELLOW
            } else if self.white_selection == i {
                Color::GREEN
            } else {
                Color::GRAY
            };
            
            let white_text = Text::new(TextFragment::new(format!("  {}", player_type.display_name()))
                .color(white_color)
                .scale(24.0));
            canvas.draw(&white_text, DrawParam::default().dest([450.0, y_offset]));
        }
        
        // Instructions
        let instructions = Text::new(TextFragment::new(
            "↑/↓: Change selection | Enter: Confirm | Tab: Switch side"
        ).color(Color::GRAY).scale(18.0));
        canvas.draw(&instructions, DrawParam::default().dest([100.0, 700.0]));
        
        Ok(())
    }
}
```

**Action Items:**
- [ ] Create `src/gui_selection.rs`
- [ ] Implement `SelectionScreen` struct
- [ ] Implement keyboard navigation
- [ ] Implement rendering logic
- [ ] Test selection flow
- [ ] Add `pub mod gui_selection;` to `src/lib.rs`

### Step 3.2: Integrate Selection into GUI Main

**File: `src/main_gui.rs` (modify)**

**Changes needed:**
1. Add selection screen state
2. Show selection screen before game starts
3. Transition to game after selection
4. Handle AI player moves in update loop

```rust
use ggez::event::{self, EventHandler};
use ggez::{Context, ContextBuilder, GameResult};
use othello::{Game, Player, GraphicsState, gui_selection, player_selection::PlayerConfig};

/// Main game state for GUI version
struct MainState {
    game: Game,
    graphics: GraphicsState,
    pending_flip: Option<Vec<othello::board::Position>>,
    pending_player: Option<Player>,
    black_player: Box<dyn othello::PlayerTrait>,
    white_player: Box<dyn othello::PlayerTrait>,
    ai_thinking: bool,
}

impl MainState {
    fn new(ctx: &mut Context, config: PlayerConfig) -> GameResult<Self> {
        let (black_player, white_player) = config.create_players();
        
        Ok(MainState {
            game: Game::new(),
            graphics: GraphicsState::new(ctx),
            pending_flip: None,
            pending_player: None,
            black_player,
            white_player,
            ai_thinking: false,
        })
    }
}

impl EventHandler for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        // Update graphics (animations, etc.)
        self.graphics.update(ctx, &self.game)?;

        // Check if we have a pending flip animation to start
        if let Some(positions) = self.pending_flip.take() {
            let current_time = ctx.time.time_since_start().as_secs_f64();
            if let Some(from_player) = self.pending_player.take() {
                let to_player = from_player.opposite();
                for pos in positions {
                    self.graphics.add_flip_animation(pos, from_player, to_player, current_time);
                }
            }
        }
        
        // Handle AI player moves
        if !self.ai_thinking 
            && matches!(self.game.get_game_state(), othello::GameState::Playing)
            && !self.graphics.has_animations() 
        {
            let current_player = self.game.current_player();
            let player = match current_player {
                Player::Black => &*self.black_player,
                Player::White => &*self.white_player,
            };
            
            // Check if current player is AI (not human)
            if !matches!(player.get_name(), "Human" | "Player 1 (Black)" | "Player 2 (White)") {
                self.ai_thinking = true;
                
                // Get AI move (this may take time for MCTS)
                if let Some(position) = player.choose_move(&self.game) {
                    if let Ok(flipped) = self.game.make_move_with_flipped(position) {
                        self.pending_flip = Some(flipped);
                        self.pending_player = Some(current_player);
                    }
                } else {
                    // AI has no valid moves, skip turn
                    self.game.skip_turn().ok();
                }
                
                self.ai_thinking = false;
            }
        }

        Ok(())
    }
    
    // ... rest of EventHandler implementation
}

/// Wrapper state that handles selection screen
struct AppState {
    selection: Option<gui_selection::SelectionScreen>,
    game: Option<MainState>,
}

impl EventHandler for AppState {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        if let Some(selection) = &mut self.selection {
            if selection.is_complete() {
                if let Some(config) = selection.get_config() {
                    self.game = Some(MainState::new(ctx, config)?);
                    self.selection = None;
                }
            }
        } else if let Some(game) = &mut self.game {
            game.update(ctx)?;
        }
        Ok(())
    }
    
    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        if let Some(selection) = &mut self.selection {
            let mut canvas = ggez::graphics::Canvas::from_frame(ctx, ggez::graphics::Color::BLACK);
            selection.draw(&mut canvas, ctx)?;
            canvas.finish(ctx)?;
        } else if let Some(game) = &mut self.game {
            game.draw(ctx)?;
        }
        Ok(())
    }
    
    fn key_down_event(&mut self, _ctx: &mut Context, input: ggez::input::keyboard::KeyInput, _repeat: bool) -> GameResult {
        if let Some(selection) = &mut self.selection {
            selection.handle_key(input);
        }
        Ok(())
    }
    
    fn mouse_button_down_event(&mut self, ctx: &mut Context, button: ggez::input::mouse::MouseButton, x: f32, y: f32) -> GameResult {
        if let Some(game) = &mut self.game {
            game.mouse_button_down_event(ctx, button, x, y)?;
        }
        Ok(())
    }
}

pub fn main() -> GameResult {
    let (mut ctx, event_loop) = ContextBuilder::new("Othello", "Othello Game")
        .window_setup(ggez::conf::WindowSetup::default().title("Othello"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(800.0, 900.0))
        .build()?;

    let state = AppState {
        selection: Some(gui_selection::SelectionScreen::new()),
        game: None,
    };
    
    event::run(ctx, event_loop, state)
}
```

**Action Items:**
- [ ] Refactor `MainState` to store player instances
- [ ] Add AI move handling in `update()` method
- [ ] Create `AppState` wrapper for selection screen
- [ ] Implement state transition from selection to game
- [ ] Test with different player combinations
- [ ] Verify AI players work correctly
- [ ] Handle edge cases (no valid moves, game over)

---

## Phase 4: Enhanced Features

**Estimated Time:** 2-3 hours  
**Goal:** Add polish and additional features.

### Step 4.1: Add Player Name Customization (Optional)

**Enhancement:** Allow users to enter custom names for players.

**Console version:**
- Add prompt for player names after type selection
- Update `PlayerConfig` to store custom names

**GUI version:**
- Add text input fields for names
- Handle text input in selection screen

**Action Items:**
- [ ] Add name input to console selection
- [ ] Add name input to GUI selection (optional, can be deferred)
- [ ] Update `PlayerConfig` to use custom names
- [ ] Test with custom names

### Step 4.2: Add Selection Persistence (Optional)

**Enhancement:** Remember last player selection.

**Implementation:**
- Save selection to a config file (e.g., `player_config.json`)
- Load on startup
- Provide option to skip selection and use saved config

**Action Items:**
- [ ] Add `serde` dependency for serialization
- [ ] Implement save/load for `PlayerConfig`
- [ ] Add "Use last selection" option
- [ ] Test persistence

### Step 4.3: Add Visual Feedback for AI Thinking

**Enhancement:** Show indicator when AI is thinking.

**GUI version:**
- Display "AI is thinking..." message
- Show progress indicator for MCTS (optional)

**Console version:**
- Print "AI is thinking..." message
- Show dots or spinner

**Action Items:**
- [ ] Add thinking indicator to GUI
- [ ] Add thinking message to console
- [ ] Test with slow AI (MCTS Expert)

---

## Phase 5: Testing and Validation

**Estimated Time:** 2-3 hours  
**Goal:** Comprehensive testing of player selection.

### Test Cases

**Console Version:**
- [ ] Test all player type selections
- [ ] Test invalid input handling
- [ ] Test Human vs Human gameplay
- [ ] Test Human vs AI gameplay
- [ ] Test Human vs MCTS (all difficulties) gameplay
- [ ] Test AI vs AI gameplay
- [ ] Test MCTS vs MCTS gameplay
- [ ] Verify AI players make moves automatically
- [ ] Verify game flow works correctly

**GUI Version:**
- [ ] Test selection screen navigation
- [ ] Test all player type combinations
- [ ] Test keyboard input (arrows, enter, tab)
- [ ] Test transition from selection to game
- [ ] Test AI move handling
- [ ] Test visual feedback
- [ ] Test with different window sizes (if applicable)

**Edge Cases:**
- [ ] Test with no valid moves (AI should skip)
- [ ] Test game over detection
- [ ] Test rapid clicking during AI thinking
- [ ] Test switching between selection and game

### Action Items:
- [ ] Create test checklist
- [ ] Test all combinations systematically
- [ ] Fix any bugs discovered
- [ ] Document known limitations

---

## Phase 6: Documentation and Polish

**Estimated Time:** 1-2 hours  
**Goal:** Update documentation and add final polish.

### Step 6.1: Update README

**File: `README.md` (modify)**

Add section about player selection:

```markdown
## Player Selection

Both the GUI and console versions now support selecting different player types:

- **Human**: Interactive player (mouse clicks in GUI, keyboard input in console)
- **AI (Random)**: Makes random valid moves
- **MCTS (Easy/Medium/Hard/Expert)**: Monte Carlo Tree Search AI with varying difficulty

### Console Version
When you start the game, you'll be prompted to select player types for both Black and White.

### GUI Version
A selection screen appears on startup. Use arrow keys to navigate, Enter to confirm, and Tab to switch between Black and White selection.
```

**Action Items:**
- [ ] Update README with player selection information
- [ ] Add screenshots or examples (if applicable)
- [ ] Document keyboard controls for GUI

### Step 6.2: Code Documentation

**Action Items:**
- [ ] Add rustdoc comments to new modules
- [ ] Document public functions and types
- [ ] Add usage examples
- [ ] Run `cargo doc` to verify

### Step 6.3: Final Polish

**Action Items:**
- [ ] Run `cargo clippy` and fix warnings
- [ ] Format code with `cargo fmt`
- [ ] Verify all tests pass
- [ ] Test on different platforms (if applicable)

---

## Implementation Timeline

| Phase | Estimated Time | Cumulative Time |
|-------|---------------|-----------------|
| Phase 1: Data Structures | 1-2 hours | 1-2 hours |
| Phase 2: Console Selection | 2-3 hours | 3-5 hours |
| Phase 3: GUI Selection | 4-6 hours | 7-11 hours |
| Phase 4: Enhanced Features | 2-3 hours | 9-14 hours |
| Phase 5: Testing | 2-3 hours | 11-17 hours |
| Phase 6: Documentation | 1-2 hours | 12-19 hours |

**Total Estimated Time:** 12-19 hours (approximately 2-3 days of work)

---

## Dependencies

### New Dependencies
- None required (all functionality uses existing code)

### Optional Dependencies (for Phase 4.2)
- `serde` with `derive` feature - For config persistence
- `serde_json` - For JSON serialization

---

## File Structure After Implementation

```
src/
├── main.rs                    # Console version (modified)
├── main_gui.rs                # GUI version (modified)
├── player_selection.rs        # NEW: Player selection types
├── console_selection.rs       # NEW: Console selection UI
├── gui_selection.rs           # NEW: GUI selection screen
├── player.rs                  # (existing)
├── human_player.rs            # (existing)
├── ai_player.rs               # (existing)
├── mcts/
│   └── player.rs              # (existing)
└── ... (other existing files)
```

---

## Common Pitfalls and Solutions

### Issue: AI Moves Too Fast in GUI

**Problem:** AI makes moves before animations complete.

**Solution:** Check `graphics.has_animations()` before processing AI moves.

### Issue: Player Trait Object Lifetime

**Problem:** Storing `Box<dyn PlayerTrait>` in game state.

**Solution:** Ensure players outlive the game state, or use `'static` bound if needed.

### Issue: Selection Screen Not Showing

**Problem:** State transition not working correctly.

**Solution:** Verify `AppState` properly manages selection and game states.

### Issue: Console Input Parsing

**Problem:** Invalid input causes crashes or infinite loops.

**Solution:** Use robust input parsing with error handling and retry logic.

---

## Future Enhancements

1. **Mouse Selection in GUI**: Click on player types instead of keyboard navigation
2. **Player Statistics**: Track wins/losses for different player types
3. **Tournament Mode**: Play multiple games with different player combinations
4. **Custom MCTS Settings**: Allow users to configure iterations, exploration constant, etc.
5. **Player Profiles**: Save favorite player combinations
6. **Replay Mode**: Watch AI vs AI games automatically

---

## Revision History

- **v1.0** (Initial): Complete implementation plan created


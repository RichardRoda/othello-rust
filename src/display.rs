use crate::board::{Board, Cell, Position};
use crate::game::{Game, GameState, Player};

/// ANSI color codes for terminal output
mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    
    // Colors
    pub const YELLOW: &str = "\x1b[33m";
    pub const CYAN: &str = "\x1b[36m";
    
    
    // Bright colors
    pub const BRIGHT_BLACK: &str = "\x1b[90m";
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const BRIGHT_MAGENTA: &str = "\x1b[95m";
    pub const BRIGHT_WHITE: &str = "\x1b[97m"; 
}

/// Color configuration
struct ColorConfig {
    enabled: bool,
}

impl ColorConfig {
    fn new() -> Self {
        // Check if we're in a terminal that supports colors
        // On Windows, colors may not work in some terminals, but we'll try anyway
        let enabled = std::env::var("NO_COLOR").is_err() && atty::is(atty::Stream::Stdout);
        ColorConfig { enabled }
    }
    
    fn colorize(&self, color: &str, text: &str) -> String {
        if self.enabled {
            format!("{}{}{}", color, text, colors::RESET)
        } else {
            text.to_string()
        }
    }
    
    fn bold(&self, text: &str) -> String {
        if self.enabled {
            format!("{}{}{}", colors::BOLD, text, colors::RESET)
        } else {
            text.to_string()
        }
    }
}

lazy_static::lazy_static! {
    static ref COLOR_CONFIG: ColorConfig = ColorConfig::new();
}

/// Render the entire game state
pub fn render_game(game: &Game) {
    clear_screen();
    render_board_with_valid_moves(game.get_board(), &game.get_valid_moves());
    render_score(game);
    render_turn(game);
    render_valid_moves(&game.get_valid_moves());
}

/// Render the game board with optional valid move highlighting
pub fn render_board(board: &Board) {
    render_board_with_valid_moves(board, &[]);
}

/// Render the game board with highlighted valid moves
pub fn render_board_with_valid_moves(board: &Board, valid_moves: &[Position]) {
    // Header with column labels
    print!("\n{}", COLOR_CONFIG.bold("     "));
    for col in 0..8 {
        let letter = (b'A' + col as u8) as char;
        print!("{} ", COLOR_CONFIG.colorize(colors::CYAN, &letter.to_string()));
    }
    println!();
    
    // Top border
    println!("{}", COLOR_CONFIG.colorize(colors::BRIGHT_BLACK, "   ┌─┬─┬─┬─┬─┬─┬─┬─┐"));
    
    for row in 0..8 {
        // Row number with left border
        print!("{} ", COLOR_CONFIG.colorize(colors::CYAN, &format!("{:2}", row + 1)));
        print!("{}", COLOR_CONFIG.colorize(colors::BRIGHT_BLACK, "│"));
        
        for col in 0..8 {
            let pos = Position::new(row, col);
            let cell = board.get_cell(pos).unwrap_or(Cell::Empty);
            let is_valid_move = valid_moves.contains(&pos);
            
            // Format cell with optional valid move highlighting
            let cell_display = if is_valid_move && cell == Cell::Empty {
                // Highlight valid moves with a green dot (only if empty)
                COLOR_CONFIG.colorize(colors::BRIGHT_GREEN, "•")
            } else {
                format_cell(cell)
            };
            
            print!("{}", cell_display);
            print!("{}", COLOR_CONFIG.colorize(colors::BRIGHT_BLACK, "│"));
        }
        println!();
        
        // Horizontal separator (except after last row)
        if row < 7 {
            println!("{}", COLOR_CONFIG.colorize(colors::BRIGHT_BLACK, "   ├─┼─┼─┼─┼─┼─┼─┼─┤"));
        }
    }
    
    // Bottom border
    println!("{}", COLOR_CONFIG.colorize(colors::BRIGHT_BLACK, "   └─┴─┴─┴─┴─┴─┴─┴─┘"));
    println!();
}

/// Format a cell for display with colors
fn format_cell(cell: Cell) -> String {
    match cell {
        Cell::Empty => COLOR_CONFIG.colorize(colors::BRIGHT_BLACK, "·"),
        Cell::Black => COLOR_CONFIG.colorize(colors::BRIGHT_BLACK, "●"), // Filled circle
        Cell::White => COLOR_CONFIG.colorize(colors::BRIGHT_WHITE, "○"), // Empty circle
    }
}

/// Render the current score with colors
pub fn render_score(game: &Game) {
    let (black, white) = game.get_score();
    
    let black_symbol = COLOR_CONFIG.colorize(colors::BRIGHT_BLACK, "●");
    let white_symbol = COLOR_CONFIG.colorize(colors::BRIGHT_WHITE, "○");
    
    println!("{}", COLOR_CONFIG.bold("Score:"));
    println!("  Black ({}): {}", black_symbol, black);
    println!("  White ({}): {}", white_symbol, white);
    println!();
}

/// Render whose turn it is with color
pub fn render_turn(game: &Game) {
    let player_str = match game.current_player() {
        Player::Black => COLOR_CONFIG.colorize(colors::BRIGHT_BLACK, "● Black"),
        Player::White => COLOR_CONFIG.colorize(colors::BRIGHT_WHITE, "○ White"),
    };
    
    println!("{}", COLOR_CONFIG.bold(&format!("Current Player: {}", player_str)));
    println!();
}

/// Render valid moves with formatting
pub fn render_valid_moves(moves: &[Position]) {
    if moves.is_empty() {
        println!("{}", COLOR_CONFIG.colorize(colors::YELLOW, "No valid moves available"));
    } else {
        let move_str = moves
            .iter()
            .map(|pos| {
                let col_letter = (b'a' + pos.col as u8) as char;
                format!("{}{}", col_letter, pos.row + 1)
            })
            .collect::<Vec<_>>()
            .join(", ");
        
        println!("{} {}", 
                 COLOR_CONFIG.bold("Valid moves:"),
                 COLOR_CONFIG.colorize(colors::BRIGHT_GREEN, &move_str));
    }
    println!();
}

/// Render game over screen with enhanced formatting
pub fn render_game_over(game: &Game) {
    clear_screen();
    render_board(game.get_board());
    render_score(game);
    
    println!("{}", COLOR_CONFIG.colorize(colors::BRIGHT_MAGENTA, "╔════════════════════════╗"));
    println!("{}", COLOR_CONFIG.colorize(colors::BRIGHT_MAGENTA, "║   === GAME OVER ===    ║"));
    println!("{}", COLOR_CONFIG.colorize(colors::BRIGHT_MAGENTA, "╚════════════════════════╝"));
    println!();
    
    match game.get_game_state() {
        GameState::GameOver { winner } => {
            match winner {
                Some(Player::Black) => {
                    println!("{}", COLOR_CONFIG.bold(&COLOR_CONFIG.colorize(
                        colors::BRIGHT_BLACK, 
                        "● Black wins!"
                    )));
                }
                Some(Player::White) => {
                    println!("{}", COLOR_CONFIG.bold(&COLOR_CONFIG.colorize(
                        colors::BRIGHT_WHITE, 
                        "○ White wins!"
                    )));
                }
                None => {
                    println!("{}", COLOR_CONFIG.bold(&COLOR_CONFIG.colorize(
                        colors::YELLOW, 
                        "It's a tie!"
                    )));
                }
            }
        }
        GameState::Playing => {
            println!("{}", COLOR_CONFIG.colorize(colors::YELLOW, "Game is still in progress"));
        }
    }
    
    let (black, white) = game.get_score();
    println!();
    println!("{}", COLOR_CONFIG.bold("Final Score:"));
    
    let black_score = COLOR_CONFIG.colorize(colors::BRIGHT_BLACK, &format!("● Black: {}", black));
    let white_score = COLOR_CONFIG.colorize(colors::BRIGHT_WHITE, &format!("○ White: {}", white));
    
    println!("  {}", black_score);
    println!("  {}", white_score);
    println!();
}

/// Clear the screen (works on most terminals)
pub fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H");
}

/// Display a prompt
pub fn render_prompt() {
    print!("{}", COLOR_CONFIG.colorize(colors::CYAN, "> "));
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

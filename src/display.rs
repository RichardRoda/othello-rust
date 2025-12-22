use crate::board::{Board, Cell};
use crate::game::{Game, GameState, Player};

/// Render the entire game state
pub fn render_game(game: &Game) {
    clear_screen();
    render_board(game.get_board());
    render_score(game);
    render_turn(game);
    render_valid_moves(&game.get_valid_moves());
}

/// Render the game board
pub fn render_board(board: &Board) {
    println!("\n    A B C D E F G H");
    
    for row in 0..8 {
        print!("{:2} ", row + 1);
        for col in 0..8 {
            let cell = board.get_cell(crate::board::Position::new(row, col)).unwrap_or(Cell::Empty);
            print!("{} ", format_cell(cell));
        }
        println!();
    }
    println!();
}

/// Format a cell for display
fn format_cell(cell: Cell) -> char {
    match cell {
        Cell::Empty => '.',
        Cell::Black => 'B',
        Cell::White => 'W',
    }
}

/// Render the current score
pub fn render_score(game: &Game) {
    let (black, white) = game.get_score();
    println!("Black: {}  White: {}", black, white);
}

/// Render whose turn it is
pub fn render_turn(game: &Game) {
    println!("Current Player: {}", game.current_player());
}

/// Render valid moves
pub fn render_valid_moves(moves: &[crate::board::Position]) {
    if moves.is_empty() {
        println!("No valid moves available");
    } else {
        let move_str = moves
            .iter()
            .map(|pos| {
                let col_letter = (b'a' + pos.col as u8) as char;
                format!("{}{}", col_letter, pos.row + 1)
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("Valid moves: {}", move_str);
    }
}

/// Render game over screen
pub fn render_game_over(game: &Game) {
    clear_screen();
    render_board(game.get_board());
    render_score(game);
    
    println!("\n=== GAME OVER ===");
    
    match game.get_game_state() {
        GameState::GameOver { winner } => {
            match winner {
                Some(Player::Black) => println!("Black wins!"),
                Some(Player::White) => println!("White wins!"),
                None => println!("It's a tie!"),
            }
        }
        GameState::Playing => {
            println!("Game is still in progress");
        }
    }
    
    let (black, white) = game.get_score();
    println!("\nFinal score:");
    println!("Black: {}", black);
    println!("White: {}", white);
}

/// Clear the screen (works on most terminals)
pub fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H");
}

/// Display a prompt
pub fn render_prompt() {
    print!("> ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
}


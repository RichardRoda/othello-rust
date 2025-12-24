use crate::board::Position;
use crate::game::Game;
use crate::player::PlayerTrait;

/// Human player that takes input from stdin
pub struct HumanPlayer {
    name: String,
}

impl HumanPlayer {
    /// Create a new human player
    pub fn new(name: impl Into<String>) -> Self {
        HumanPlayer {
            name: name.into(),
        }
    }

    /// Parse input string to a position
    /// Accepts formats: "a1", "A1", "1,1", "1 1", "0,0"
    fn parse_input(&self, input: &str) -> Option<Position> {
        let input = input.trim().to_lowercase();

        // Try letter-number format (e.g., "a1", "h8")
        if input.len() >= 2 {
            let mut chars = input.chars();
            if let Some(letter) = chars.next() {
                if letter >= 'a' && letter <= 'h' {
                    if let Ok(number) = input[1..].parse::<usize>() {
                        if number >= 1 && number <= 8 {
                            return Some(Position::new(number - 1, (letter as u8 - b'a') as usize));
                        }
                    }
                }
            }
        }

        // Try comma-separated format (e.g., "3,4", "0,0")
        if let Some(comma_pos) = input.find(',') {
            if let (Ok(row), Ok(col)) = (
                input[..comma_pos].trim().parse::<usize>(),
                input[comma_pos + 1..].trim().parse::<usize>(),
            ) {
                if row < 8 && col < 8 {
                    return Some(Position::new(row, col));
                }
            }
        }

        // Try space-separated format (e.g., "3 4", "0 0")
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() == 2 {
            if let (Ok(row), Ok(col)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                if row < 8 && col < 8 {
                    return Some(Position::new(row, col));
                }
            }
        }

        None
    }
}

impl PlayerTrait for HumanPlayer {
    fn choose_move(&self, game: &Game) -> Option<Position> {
        let valid_moves = game.get_valid_moves();
        
        loop {
            println!("\n{}'s turn ({}). Enter your move (e.g., 'd3' or '3,3'):", 
                     self.get_name(), 
                     game.current_player());
            println!("Valid moves: {}", format_moves(&valid_moves));
            println!("Or 'q' to quit");

            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                println!("Error reading input. Please try again.");
                continue;
            }

            let input = input.trim();
            if input.eq_ignore_ascii_case("q") || input.eq_ignore_ascii_case("quit") {
                return None;
            }

            if let Some(position) = self.parse_input(input) {
                // Verify it's a valid move
                if valid_moves.contains(&position) {
                    return Some(position);
                } else {
                    println!("That's not a valid move. Please choose from the valid moves listed above.");
                }
            } else {
                println!("Invalid input format. Please use format like 'd3' or '3,3' (row, col)");
            }
        }
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Format a list of moves for display
fn format_moves(moves: &[Position]) -> String {
    if moves.is_empty() {
        return "none".to_string();
    }

    moves
        .iter()
        .map(|pos| {
            let col_letter = (b'a' + pos.col as u8) as char;
            format!("{}{}", col_letter, pos.row + 1)
        })
        .collect::<Vec<_>>()
        .join(", ")
}




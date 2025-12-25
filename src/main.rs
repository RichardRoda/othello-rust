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
        // Render the game state
        othello::display::render_game(&game);
        
        // Get the current player
        let player: &dyn PlayerTrait = match game.current_player() {
            othello::Player::Black => black_player.as_ref(),
            othello::Player::White => white_player.as_ref(),
        };
        
        // Check if current player has valid moves
        let valid_moves = game.get_valid_moves();
        if valid_moves.is_empty() {
            println!("\n{} has no valid moves. Skipping turn.", player.get_name());
            
            if let Err(e) = game.skip_turn() {
                eprintln!("Error skipping turn: {}", e);
                break;
            }
        } else {
            // Get move from player
            match player.choose_move(&game) {
                Some(position) => {
                    match game.make_move(position) {
                        Ok(()) => {
                            // Move successful, continue to next turn
                        }
                        Err(e) => {
                            eprintln!("Error making move: {}", e);
                            println!("Press Enter to continue...");
                            let _ = std::io::stdin().read_line(&mut String::new());
                        }
                    }
                }
                None => {
                    // Player chose to quit
                    println!("\n{} has quit the game.", player.get_name());
                    break;
                }
            }
        }
        
        // Check if game is over
        match game.get_game_state() {
            othello::GameState::Playing => {
                // Continue playing
            }
            othello::GameState::GameOver { .. } => {
                othello::display::render_game_over(&game);
                break;
            }
        }
    }
    
    println!("\nThanks for playing!");
}


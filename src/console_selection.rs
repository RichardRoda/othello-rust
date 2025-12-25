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


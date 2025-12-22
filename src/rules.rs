use crate::board::{Board, Cell, Position};
use crate::error::GameError;
use crate::game::Player;

/// Direction for move validation and piece flipping
#[derive(Debug, Clone, Copy)]
enum Direction {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

impl Direction {
    /// Get all 8 directions
    fn all() -> [Direction; 8] {
        [
            Direction::North,
            Direction::NorthEast,
            Direction::East,
            Direction::SouthEast,
            Direction::South,
            Direction::SouthWest,
            Direction::West,
            Direction::NorthWest,
        ]
    }

    /// Get the next position in this direction
    fn apply(&self, pos: Position) -> Option<Position> {
        match self {
            Direction::North => {
                if pos.row > 0 {
                    Some(Position::new(pos.row - 1, pos.col))
                } else {
                    None
                }
            }
            Direction::NorthEast => {
                if pos.row > 0 && pos.col < 7 {
                    Some(Position::new(pos.row - 1, pos.col + 1))
                } else {
                    None
                }
            }
            Direction::East => {
                if pos.col < 7 {
                    Some(Position::new(pos.row, pos.col + 1))
                } else {
                    None
                }
            }
            Direction::SouthEast => {
                if pos.row < 7 && pos.col < 7 {
                    Some(Position::new(pos.row + 1, pos.col + 1))
                } else {
                    None
                }
            }
            Direction::South => {
                if pos.row < 7 {
                    Some(Position::new(pos.row + 1, pos.col))
                } else {
                    None
                }
            }
            Direction::SouthWest => {
                if pos.row < 7 && pos.col > 0 {
                    Some(Position::new(pos.row + 1, pos.col - 1))
                } else {
                    None
                }
            }
            Direction::West => {
                if pos.col > 0 {
                    Some(Position::new(pos.row, pos.col - 1))
                } else {
                    None
                }
            }
            Direction::NorthWest => {
                if pos.row > 0 && pos.col > 0 {
                    Some(Position::new(pos.row - 1, pos.col - 1))
                } else {
                    None
                }
            }
        }
    }
}

/// Check if a move is valid for the given player
pub fn is_valid_move(board: &Board, position: Position, player: Player) -> bool {
    // Check if position is valid
    if !board.is_valid_position(position) {
        return false;
    }

    // Check if cell is empty
    if let Ok(cell) = board.get_cell(position) {
        if cell != Cell::Empty {
            return false;
        }
    } else {
        return false;
    }

    // Check each direction for a valid flank
    for direction in Direction::all() {
        if check_direction(board, position, direction, player) {
            return true;
        }
    }

    false
}

/// Check if a direction contains a valid flank
fn check_direction(
    board: &Board,
    position: Position,
    direction: Direction,
    player: Player,
) -> bool {
    let opponent = player.opposite();
    let opponent_cell = Cell::from_player(opponent);
    let player_cell = Cell::from_player(player);

    let mut current = match direction.apply(position) {
        Some(pos) => pos,
        None => return false,
    };

    let mut found_opponent = false;

    // Traverse in the direction
    while board.is_valid_position(current) {
        match board.get_cell(current) {
            Ok(Cell::Empty) => return false,
            Ok(cell) if cell == opponent_cell => {
                found_opponent = true;
                current = match direction.apply(current) {
                    Some(pos) => pos,
                    None => return false,
                };
            }
            Ok(cell) if cell == player_cell => {
                return found_opponent;
            }
            _ => return false,
        }
    }

    false
}

/// Get all valid moves for a player
pub fn get_valid_moves(board: &Board, player: Player) -> Vec<Position> {
    let mut moves = Vec::new();

    for row in 0..8 {
        for col in 0..8 {
            let pos = Position::new(row, col);
            if is_valid_move(board, pos, player) {
                moves.push(pos);
            }
        }
    }

    moves
}

/// Check if a player has any valid moves
pub fn has_valid_move(board: &Board, player: Player) -> bool {
    !get_valid_moves(board, player).is_empty()
}

/// Apply a move and flip pieces
pub fn apply_move(
    board: &mut Board,
    position: Position,
    player: Player,
) -> Result<Vec<Position>, GameError> {
    if !is_valid_move(board, position, player) {
        return Err(GameError::InvalidMove);
    }

    let player_cell = Cell::from_player(player);
    let mut flipped_positions = Vec::new();

    // Set the position
    board.set_cell(position, player_cell)?;

    // Flip pieces in all valid directions
    for direction in Direction::all() {
        let flipped = get_flipped_in_direction(board, position, direction, player);
        for pos in &flipped {
            board.set_cell(*pos, player_cell)?;
        }
        flipped_positions.extend(flipped);
    }

    Ok(flipped_positions)
}

/// Get positions that would be flipped in a given direction
fn get_flipped_in_direction(
    board: &Board,
    position: Position,
    direction: Direction,
    player: Player,
) -> Vec<Position> {
    let opponent = player.opposite();
    let opponent_cell = Cell::from_player(opponent);
    let player_cell = Cell::from_player(player);

    let mut result = Vec::new();
    let mut current = match direction.apply(position) {
        Some(pos) => pos,
        None => return result,
    };

    // Collect opponent pieces
    while board.is_valid_position(current) {
        match board.get_cell(current) {
            Ok(Cell::Empty) => return Vec::new(),
            Ok(cell) if cell == opponent_cell => {
                result.push(current);
                current = match direction.apply(current) {
                    Some(pos) => pos,
                    None => return Vec::new(),
                };
            }
            Ok(cell) if cell == player_cell => {
                return result; // Valid flank found
            }
            _ => return Vec::new(),
        }
    }

    Vec::new() // No valid flank
}

/// Check if the game is over
pub fn is_game_over(board: &Board) -> bool {
    !has_valid_move(board, Player::Black) && !has_valid_move(board, Player::White)
}

/// Get the winner (None for a tie)
pub fn get_winner(board: &Board) -> Option<Player> {
    let black_count = board.count_pieces(Player::Black);
    let white_count = board.count_pieces(Player::White);

    match black_count.cmp(&white_count) {
        std::cmp::Ordering::Greater => Some(Player::Black),
        std::cmp::Ordering::Less => Some(Player::White),
        std::cmp::Ordering::Equal => None,
    }
}


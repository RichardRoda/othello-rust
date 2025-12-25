use crate::error::GameError;
use crate::game::Player;

/// Represents the state of a cell on the board
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    Black,
    White,
}

impl Cell {
    /// Convert a Player to a Cell
    pub fn from_player(player: Player) -> Self {
        match player {
            Player::Black => Cell::Black,
            Player::White => Cell::White,
        }
    }

    /// Convert a Cell to a Player (returns None if Empty)
    pub fn to_player(self) -> Option<Player> {
        match self {
            Cell::Empty => None,
            Cell::Black => Some(Player::Black),
            Cell::White => Some(Player::White),
        }
    }
}

impl std::fmt::Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cell::Empty => write!(f, "."),
            Cell::Black => write!(f, "B"),
            Cell::White => write!(f, "W"),
        }
    }
}

/// Position on the board (0-indexed)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub row: usize,
    pub col: usize,
}

impl Position {
    /// Create a new position
    pub fn new(row: usize, col: usize) -> Self {
        Position { row, col }
    }
}

/// The game board (8x8 grid)
#[derive(Debug, Clone)]
pub struct Board {
    grid: [[Cell; 8]; 8],
}

impl Board {
    /// Create a new board with initial setup (4 pieces in center)
    pub fn new() -> Self {
        let mut board = Board {
            grid: [[Cell::Empty; 8]; 8],
        };
        
        // Initial setup: 2x2 square in center
        board.grid[3][3] = Cell::White;
        board.grid[3][4] = Cell::Black;
        board.grid[4][3] = Cell::Black;
        board.grid[4][4] = Cell::White;
        
        board
    }

    /// Get the cell state at the given position
    pub fn get_cell(&self, pos: Position) -> Result<Cell, GameError> {
        if !self.is_valid_position(pos) {
            return Err(GameError::OutOfBounds);
        }
        Ok(self.grid[pos.row][pos.col])
    }

    /// Set the cell state at the given position
    pub fn set_cell(&mut self, pos: Position, cell: Cell) -> Result<(), GameError> {
        if !self.is_valid_position(pos) {
            return Err(GameError::OutOfBounds);
        }
        self.grid[pos.row][pos.col] = cell;
        Ok(())
    }

    /// Check if a position is valid (within board bounds)
    pub fn is_valid_position(&self, pos: Position) -> bool {
        pos.row < 8 && pos.col < 8
    }

    /// Count pieces of a given color
    pub fn count_pieces(&self, player: Player) -> usize {
        let cell = Cell::from_player(player);
        self.grid
            .iter()
            .flatten()
            .filter(|&&c| c == cell)
            .count()
    }

    /// Get a read-only reference to the grid
    pub fn grid(&self) -> &[[Cell; 8]; 8] {
        &self.grid
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}




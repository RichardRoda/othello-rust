use ggez::graphics::{Canvas, Color, DrawMode, DrawParam, Mesh, Rect, Text, TextFragment};
use ggez::{Context, GameResult};
use keyframe::{ease, functions::EaseInOutCubic};
use crate::board::{Cell, Position};
use crate::game::{Game, GameState, Player};

/// Colors for the game
mod colors {
    use ggez::graphics::Color;

    pub const BOARD_GREEN: Color = Color {
        r: 0.0,
        g: 0.5,
        b: 0.0,
        a: 1.0,
    };
    
    pub const BOARD_LIGHT: Color = Color {
        r: 0.2,
        g: 0.7,
        b: 0.2,
        a: 1.0,
    };
    
    pub const BOARD_DARK: Color = Color {
        r: 0.0,
        g: 0.4,
        b: 0.0,
        a: 1.0,
    };

    pub const BLACK_PIECE: Color = Color {
        r: 0.1,
        g: 0.1,
        b: 0.1,
        a: 1.0,
    };

    pub const WHITE_PIECE: Color = Color {
        r: 0.95,
        g: 0.95,
        b: 0.95,
        a: 1.0,
    };

    pub const VALID_MOVE: Color = Color {
        r: 0.0,
        g: 0.8,
        b: 0.3,
        a: 0.5,
    };

    pub const TEXT_COLOR: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
}

/// Animation state for a flipping piece
#[derive(Clone)]
struct FlipAnimation {
    position: Position,
    start_time: f64,
    duration: f64, // Duration in seconds
    from_player: Player,
    to_player: Player,
}

impl FlipAnimation {
    fn new(position: Position, from_player: Player, to_player: Player, start_time: f64) -> Self {
        FlipAnimation {
            position,
            start_time,
            duration: 0.3, // 300ms animation
            from_player,
            to_player,
        }
    }

    fn progress(&self, current_time: f64) -> f32 {
        let elapsed = current_time - self.start_time;
        if elapsed >= self.duration {
            1.0
        } else {
            let t = (elapsed / self.duration) as f32;
            // Use ease_in_out_cubic for smooth animation
            ease(EaseInOutCubic, t, 0.0, 1.0)
        }
    }

    fn is_complete(&self, current_time: f64) -> bool {
        current_time - self.start_time >= self.duration
    }
}

/// Main graphics state
pub struct GraphicsState {
    board_size: f32,
    cell_size: f32,
    piece_radius: f32,
    board_offset_x: f32,
    board_offset_y: f32,
    flip_animations: Vec<FlipAnimation>,
    current_time: f64,
}

impl GraphicsState {
    pub fn new(ctx: &Context) -> Self {
        let (screen_w, screen_h) = ctx.gfx.size();
        let board_size = screen_w.min(screen_h - 150.0) * 0.9;
        let cell_size = board_size / 8.0;
        let piece_radius = cell_size * 0.4;
        
        GraphicsState {
            board_size,
            cell_size,
            piece_radius,
            board_offset_x: (screen_w - board_size) / 2.0,
            board_offset_y: 50.0,
            flip_animations: Vec::new(),
            current_time: 0.0,
        }
    }

    /// Check if any animations are running
    pub fn has_animations(&self) -> bool {
        !self.flip_animations.is_empty()
    }

    /// Update animation state
    pub fn update(&mut self, ctx: &mut Context, _game: &Game) -> GameResult {
        self.current_time = ctx.time.time_since_start().as_secs_f64();
        
        // Remove completed animations
        self.flip_animations.retain(|anim| !anim.is_complete(self.current_time));

        // Recalculate board size on window resize
        let (screen_w, screen_h) = ctx.gfx.size();
        self.board_size = screen_w.min(screen_h - 150.0) * 0.9;
        self.cell_size = self.board_size / 8.0;
        self.piece_radius = self.cell_size * 0.4;
        self.board_offset_x = (screen_w - self.board_size) / 2.0;
        self.board_offset_y = 50.0;

        Ok(())
    }

    /// Add a flip animation
    pub fn add_flip_animation(&mut self, position: Position, from_player: Player, to_player: Player, start_time: f64) {
        self.flip_animations.push(FlipAnimation::new(position, from_player, to_player, start_time));
    }

    /// Convert screen coordinates to board position
    pub fn screen_to_position(&self, x: f32, y: f32) -> Option<Position> {
        let board_x = x - self.board_offset_x;
        let board_y = y - self.board_offset_y;

        if board_x < 0.0 || board_y < 0.0 || board_x > self.board_size || board_y > self.board_size {
            return None;
        }

        let col = (board_x / self.cell_size) as usize;
        let row = (board_y / self.cell_size) as usize;

        if row < 8 && col < 8 {
            Some(Position::new(row, col))
        } else {
            None
        }
    }

    /// Render the entire game
    pub fn draw(&mut self, ctx: &mut Context, game: &Game) -> GameResult {
        let mut canvas = Canvas::from_frame(ctx, Color::new(0.1, 0.1, 0.15, 1.0));
        
        self.draw_board(ctx, &mut canvas)?;
        self.draw_pieces(ctx, &mut canvas, game)?;
        self.draw_valid_moves(ctx, &mut canvas, game)?;
        self.draw_ui(ctx, &mut canvas, game)?;

        canvas.finish(ctx)?;
        Ok(())
    }

    /// Draw the board grid
    fn draw_board(&self, ctx: &mut Context, canvas: &mut Canvas) -> GameResult {
        let rect = Rect::new(
            self.board_offset_x,
            self.board_offset_y,
            self.board_size,
            self.board_size,
        );
        let mesh = Mesh::new_rectangle(ctx, DrawMode::fill(), rect, colors::BOARD_GREEN)?;
        canvas.draw(&mesh, DrawParam::default());

        // Draw grid lines
        let line_color = Color::new(0.0, 0.3, 0.0, 1.0);
        for i in 0..9 {
            let pos = self.board_offset_x + (i as f32 * self.cell_size);
            // Vertical lines
            let points = [
                [pos, self.board_offset_y],
                [pos, self.board_offset_y + self.board_size],
            ];
            let v_line = Mesh::new_line(ctx, &points, 2.0, line_color)?;
            canvas.draw(&v_line, DrawParam::default());

            // Horizontal lines
            let h_pos = self.board_offset_y + pos - self.board_offset_x;
            let h_points = [
                [self.board_offset_x, h_pos],
                [self.board_offset_x + self.board_size, h_pos],
            ];
            let h_line = Mesh::new_line(ctx, &h_points, 2.0, line_color)?;
            canvas.draw(&h_line, DrawParam::default());
        }

        Ok(())
    }

    /// Draw game pieces
    fn draw_pieces(&self, ctx: &mut Context, canvas: &mut Canvas, game: &Game) -> GameResult {
        let board = game.get_board();
        
        for row in 0..8 {
            for col in 0..8 {
                let pos = Position::new(row, col);
                if let Ok(cell) = board.get_cell(pos) {
                    if cell != Cell::Empty {
                        // Check if this piece is animating
                        if let Some(anim) = self.flip_animations.iter().find(|a| a.position == pos) {
                            self.draw_animating_piece(ctx, canvas, pos, anim, self.current_time)?;
                        } else {
                            let player = match cell {
                                Cell::Black => Player::Black,
                                Cell::White => Player::White,
                                _ => continue,
                            };
                            self.draw_piece(ctx, canvas, pos, player)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Draw a static piece
    fn draw_piece(&self, ctx: &mut Context, canvas: &mut Canvas, pos: Position, player: Player) -> GameResult {
        let center_x = self.board_offset_x + (pos.col as f32 + 0.5) * self.cell_size;
        let center_y = self.board_offset_y + (pos.row as f32 + 0.5) * self.cell_size;

        let color = match player {
            Player::Black => colors::BLACK_PIECE,
            Player::White => colors::WHITE_PIECE,
        };

        let circle = Mesh::new_circle(
            ctx,
            DrawMode::fill(),
            [center_x, center_y],
            self.piece_radius,
            0.1,
            color,
        )?;

        canvas.draw(&circle, DrawParam::default());

        // Draw a border for contrast
        let border_color = match player {
            Player::Black => Color::new(0.3, 0.3, 0.3, 1.0),
            Player::White => Color::new(0.7, 0.7, 0.7, 1.0),
        };
        let border = Mesh::new_circle(
            ctx,
            DrawMode::stroke(2.0),
            [center_x, center_y],
            self.piece_radius,
            0.1,
            border_color,
        )?;
        canvas.draw(&border, DrawParam::default());

        Ok(())
    }

    /// Draw an animating piece (flipping)
    fn draw_animating_piece(&self, ctx: &mut Context, canvas: &mut Canvas, pos: Position, anim: &FlipAnimation, current_time: f64) -> GameResult {
        let center_x = self.board_offset_x + (pos.col as f32 + 0.5) * self.cell_size;
        let center_y = self.board_offset_y + (pos.row as f32 + 0.5) * self.cell_size;

        let progress = anim.progress(current_time);
        
        // Scale animation: piece scales down to flat, then back up as the other color
        let scale = if progress < 0.5 {
            // Scale down from 1.0 to 0.1
            1.0 - (progress * 1.8)
        } else {
            // Scale up from 0.1 to 1.0
            0.1 + ((progress - 0.5) * 1.8)
        };

        let current_player = if progress < 0.5 {
            anim.from_player
        } else {
            anim.to_player
        };

        let color = match current_player {
            Player::Black => colors::BLACK_PIECE,
            Player::White => colors::WHITE_PIECE,
        };

        let radius = self.piece_radius * scale;
        if radius > 0.1 {
            let circle = Mesh::new_circle(
                ctx,
                DrawMode::fill(),
                [center_x, center_y],
                radius,
                0.1,
                color,
            )?;
            canvas.draw(&circle, DrawParam::default());
        }

        Ok(())
    }

    /// Draw valid move indicators
    fn draw_valid_moves(&self, ctx: &mut Context, canvas: &mut Canvas, game: &Game) -> GameResult {
        let valid_moves = game.get_valid_moves();
        
        for pos in valid_moves {
            let center_x = self.board_offset_x + (pos.col as f32 + 0.5) * self.cell_size;
            let center_y = self.board_offset_y + (pos.row as f32 + 0.5) * self.cell_size;

            let circle = Mesh::new_circle(
                ctx,
                DrawMode::fill(),
                [center_x, center_y],
                self.piece_radius * 0.3,
                0.1,
                colors::VALID_MOVE,
            )?;
            canvas.draw(&circle, DrawParam::default());
        }

        Ok(())
    }

    /// Draw UI elements (score, current player, etc.)
    fn draw_ui(&self, ctx: &mut Context, canvas: &mut Canvas, game: &Game) -> GameResult {
        let (black_score, white_score) = game.get_score();
        let screen_w = ctx.gfx.size().0;
        
        // Draw scores
        let score_text = Text::new(TextFragment {
            text: format!("Black: {}  White: {}", black_score, white_score),
            color: Some(colors::TEXT_COLOR),
            font: None,
            scale: Some(ggez::graphics::PxScale::from(24.0)),
        });
        let score_pos = [screen_w / 2.0 - 100.0, self.board_offset_y + self.board_size + 20.0];
        canvas.draw(&score_text, DrawParam::default().dest(score_pos));

        // Draw current player
        let player_text = Text::new(TextFragment {
            text: format!("Current: {}", game.current_player()),
            color: Some(colors::TEXT_COLOR),
            font: None,
            scale: Some(ggez::graphics::PxScale::from(20.0)),
        });
        let player_pos = [screen_w / 2.0 - 80.0, self.board_offset_y + self.board_size + 50.0];
        canvas.draw(&player_text, DrawParam::default().dest(player_pos));

        // Draw game over message
        if let GameState::GameOver { winner } = game.get_game_state() {
            let message = match winner {
                Some(Player::Black) => "Black Wins!",
                Some(Player::White) => "White Wins!",
                None => "It's a Tie!",
            };
            let game_over_text = Text::new(TextFragment {
                text: format!("GAME OVER - {}", message),
                color: Some(Color::new(1.0, 0.8, 0.0, 1.0)),
                font: None,
                scale: Some(ggez::graphics::PxScale::from(32.0)),
            });
            let game_over_pos = [screen_w / 2.0 - 150.0, 10.0];
            canvas.draw(&game_over_text, DrawParam::default().dest(game_over_pos));
        }

        Ok(())
    }
}


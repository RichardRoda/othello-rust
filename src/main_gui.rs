use ggez::event::{self, EventHandler};
use ggez::graphics;
use ggez::{Context, ContextBuilder, GameResult};
use othello::{Game, Player, GraphicsState};

/// Main game state for GUI version
struct MainState {
    game: Game,
    graphics: GraphicsState,
    pending_flip: Option<Vec<othello::board::Position>>,
    pending_player: Option<Player>,
}

impl MainState {
    fn new(ctx: &mut Context) -> GameResult<Self> {
        Ok(MainState {
            game: Game::new(),
            graphics: GraphicsState::new(ctx),
            pending_flip: None,
            pending_player: None,
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

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        self.graphics.draw(ctx, &self.game)?;
        Ok(())
    }

    fn mouse_button_down_event(
        &mut self,
        _ctx: &mut Context,
        _button: ggez::input::mouse::MouseButton,
        x: f32,
        y: f32,
    ) -> GameResult {
        // Only handle clicks if game is still playing
        if matches!(self.game.get_game_state(), othello::GameState::Playing) {
            // Check if animations are still running
            if self.graphics.has_animations() {
                return Ok(()); // Ignore clicks during animations
            }

            if let Some(position) = self.graphics.screen_to_position(x, y) {
                let current_player = self.game.current_player();
                
                // Try to make the move
                if let Ok(flipped) = self.game.make_move_with_flipped(position) {
                    // Store flipped positions for animation
                    self.pending_flip = Some(flipped);
                    self.pending_player = Some(current_player);
                }
            }
        }
        Ok(())
    }
}

pub fn main() -> GameResult {
    // Create a mutable context
    let (mut ctx, event_loop) = ContextBuilder::new("Othello", "Othello Game")
        .window_setup(ggez::conf::WindowSetup::default().title("Othello"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(800.0, 900.0))
        .build()?;

    // Create the game state
    let state = MainState::new(&mut ctx)?;
    
    // Run the game
    event::run(ctx, event_loop, state)
}


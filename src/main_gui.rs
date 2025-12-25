use ggez::event::{self, EventHandler};
use ggez::{Context, ContextBuilder, GameResult};
use othello::{Game, Player, GraphicsState, PlayerTrait, gui_selection, player_selection::PlayerConfig};

/// Main game state for GUI version
struct MainState {
    game: Game,
    graphics: GraphicsState,
    pending_flip: Option<Vec<othello::board::Position>>,
    pending_player: Option<Player>,
    black_player: Box<dyn PlayerTrait>,
    white_player: Box<dyn PlayerTrait>,
    black_is_human: bool,
    white_is_human: bool,
    ai_thinking: bool,
}

impl MainState {
    fn new(ctx: &mut Context, config: PlayerConfig) -> GameResult<Self> {
        let (black_player, white_player) = config.create_players();
        let black_is_human = config.black_type.is_human();
        let white_is_human = config.white_type.is_human();
        
        Ok(MainState {
            game: Game::new(),
            graphics: GraphicsState::new(ctx),
            pending_flip: None,
            pending_player: None,
            black_player,
            white_player,
            black_is_human,
            white_is_human,
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
            let is_human = match current_player {
                Player::Black => self.black_is_human,
                Player::White => self.white_is_human,
            };
            
            // Check if current player is AI (not human)
            if !is_human {
                let player: &dyn PlayerTrait = match current_player {
                    Player::Black => self.black_player.as_ref(),
                    Player::White => self.white_player.as_ref(),
                };
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
            
            // Only allow human players to make moves via mouse
            let current_player = self.game.current_player();
            let is_human = match current_player {
                Player::Black => self.black_is_human,
                Player::White => self.white_is_human,
            };
            
            if is_human {
                if let Some(position) = self.graphics.screen_to_position(x, y) {
                    // Try to make the move
                    if let Ok(flipped) = self.game.make_move_with_flipped(position) {
                        // Store flipped positions for animation
                        self.pending_flip = Some(flipped);
                        self.pending_player = Some(current_player);
                    }
                }
            }
        }
        Ok(())
    }
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
    let (ctx, event_loop) = ContextBuilder::new("Othello", "Othello Game")
        .window_setup(ggez::conf::WindowSetup::default().title("Othello"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(800.0, 900.0))
        .build()?;

    let state = AppState {
        selection: Some(gui_selection::SelectionScreen::new()),
        game: None,
    };
    
    event::run(ctx, event_loop, state)
}

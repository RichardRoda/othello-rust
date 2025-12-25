use ggez::graphics::{Canvas, Color, DrawParam, Text, TextFragment, Drawable};
use ggez::{Context, GameResult};
use ggez::input::keyboard::KeyInput;
use crate::player_selection::{PlayerType, PlayerConfig};

/// UI state for player selection screen
pub struct SelectionScreen {
    black_selection: usize,
    white_selection: usize,
    player_types: Vec<PlayerType>,
    selected_side: SelectionSide,
    config: Option<PlayerConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SelectionSide {
    Black,
    White,
    Confirmed,
}

impl SelectionScreen {
    pub fn new() -> Self {
        Self {
            black_selection: 0,
            white_selection: 0,
            player_types: PlayerType::all(),
            selected_side: SelectionSide::Black,
            config: None,
        }
    }
    
    /// Check if selection is complete
    pub fn is_complete(&self) -> bool {
        self.config.is_some()
    }
    
    /// Get the player configuration (if complete)
    pub fn get_config(&self) -> Option<PlayerConfig> {
        self.config.clone()
    }
    
    /// Handle keyboard input
    pub fn handle_key(&mut self, key: KeyInput) {
        // In ggez 0.10, KeyInput has an event field that is a KeyEvent
        // Use logical_key to get the key code
        match key.event.logical_key {
            ggez::winit::keyboard::Key::Named(named_key) => {
                match named_key {
                    ggez::winit::keyboard::NamedKey::ArrowUp => {
                        self.decrement_selection();
                    }
                    ggez::winit::keyboard::NamedKey::ArrowDown => {
                        self.increment_selection();
                    }
                    ggez::winit::keyboard::NamedKey::Enter => {
                        self.confirm_selection();
                    }
                    ggez::winit::keyboard::NamedKey::Tab => {
                        self.switch_side();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    
    fn increment_selection(&mut self) {
        match self.selected_side {
            SelectionSide::Black => {
                self.black_selection = (self.black_selection + 1) % self.player_types.len();
            }
            SelectionSide::White => {
                self.white_selection = (self.white_selection + 1) % self.player_types.len();
            }
            SelectionSide::Confirmed => {}
        }
    }
    
    fn decrement_selection(&mut self) {
        match self.selected_side {
            SelectionSide::Black => {
                if self.black_selection == 0 {
                    self.black_selection = self.player_types.len() - 1;
                } else {
                    self.black_selection -= 1;
                }
            }
            SelectionSide::White => {
                if self.white_selection == 0 {
                    self.white_selection = self.player_types.len() - 1;
                } else {
                    self.white_selection -= 1;
                }
            }
            SelectionSide::Confirmed => {}
        }
    }
    
    fn confirm_selection(&mut self) {
        match self.selected_side {
            SelectionSide::Black => {
                self.selected_side = SelectionSide::White;
            }
            SelectionSide::White => {
                // Create config and mark as complete
                self.config = Some(PlayerConfig::new(
                    self.player_types[self.black_selection],
                    self.player_types[self.white_selection],
                ));
                self.selected_side = SelectionSide::Confirmed;
            }
            SelectionSide::Confirmed => {}
        }
    }
    
    fn switch_side(&mut self) {
        match self.selected_side {
            SelectionSide::Black => self.selected_side = SelectionSide::White,
            SelectionSide::White => self.selected_side = SelectionSide::Black,
            SelectionSide::Confirmed => {}
        }
    }
    
    /// Render the selection screen
    pub fn draw(&self, canvas: &mut Canvas, ctx: &Context) -> GameResult {
        let (screen_w, screen_h) = ctx.gfx.size();
        
        // Title
        let title = Text::new(TextFragment {
            text: "Select Players".to_string(),
            color: Some(Color::WHITE),
            font: None,
            scale: Some(ggez::graphics::PxScale::from(48.0)),
        });
        let title_width = Text::dimensions(&title, ctx).w;
        canvas.draw(&title, DrawParam::default().dest([screen_w / 2.0 - title_width / 2.0, 50.0]));
        
        // Black player selection
        let black_label_text = if self.selected_side == SelectionSide::Black {
            "Black Player: (Selecting)"
        } else {
            "Black Player:"
        };
        let black_label = Text::new(TextFragment {
            text: black_label_text.to_string(),
            color: Some(if self.selected_side == SelectionSide::Black {
                Color::YELLOW
            } else {
                Color::WHITE
            }),
            font: None,
            scale: Some(ggez::graphics::PxScale::from(32.0)),
        });
        canvas.draw(&black_label, DrawParam::default().dest([100.0, 200.0]));
        
        // White player selection
        let white_label_text = if self.selected_side == SelectionSide::White {
            "White Player: (Selecting)"
        } else {
            "White Player:"
        };
        let white_label = Text::new(TextFragment {
            text: white_label_text.to_string(),
            color: Some(if self.selected_side == SelectionSide::White {
                Color::YELLOW
            } else {
                Color::WHITE
            }),
            font: None,
            scale: Some(ggez::graphics::PxScale::from(32.0)),
        });
        canvas.draw(&white_label, DrawParam::default().dest([100.0, 600.0]));
        
        // Player type options
        for (i, player_type) in self.player_types.iter().enumerate() {
            let y_offset = 250.0 + (i as f32 * 40.0);
            
            // Black selection
            let is_selected_black = self.selected_side == SelectionSide::Black 
                && self.black_selection == i;
            let black_color = if is_selected_black {
                Color::YELLOW
            } else if self.black_selection == i {
                Color::GREEN
            } else {
                Color::new(0.5, 0.5, 0.5, 1.0)
            };
            
            let black_text = Text::new(TextFragment {
                text: format!("  {}", player_type.display_name()),
                color: Some(black_color),
                font: None,
                scale: Some(ggez::graphics::PxScale::from(24.0)),
            });
            canvas.draw(&black_text, DrawParam::default().dest([150.0, y_offset]));
            
            // White selection
            let is_selected_white = self.selected_side == SelectionSide::White 
                && self.white_selection == i;
            let white_color = if is_selected_white {
                Color::YELLOW
            } else if self.white_selection == i {
                Color::GREEN
            } else {
                Color::new(0.5, 0.5, 0.5, 1.0)
            };
            
            let white_text = Text::new(TextFragment {
                text: format!("  {}", player_type.display_name()),
                color: Some(white_color),
                font: None,
                scale: Some(ggez::graphics::PxScale::from(24.0)),
            });
            canvas.draw(&white_text, DrawParam::default().dest([450.0, y_offset]));
        }
        
        // Instructions
        let instructions = Text::new(TextFragment {
            text: "↑/↓: Change selection | Enter: Confirm | Tab: Switch side".to_string(),
            color: Some(Color::new(0.5, 0.5, 0.5, 1.0)),
            font: None,
            scale: Some(ggez::graphics::PxScale::from(18.0)),
        });
        canvas.draw(&instructions, DrawParam::default().dest([100.0, screen_h - 50.0]));
        
        Ok(())
    }
}


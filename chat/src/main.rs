use eframe::egui;
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

// --- Colors (Claude Code web style) ---
const BG: egui::Color32 = egui::Color32::from_rgb(23, 23, 23);
const USER_BUBBLE: egui::Color32 = egui::Color32::from_rgb(55, 55, 55);
const ASSISTANT_BUBBLE: egui::Color32 = egui::Color32::from_rgb(40, 40, 40);
const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(236, 236, 236);
const TEXT_SECONDARY: egui::Color32 = egui::Color32::from_rgb(140, 140, 140);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(217, 119, 80);
const INPUT_BG: egui::Color32 = egui::Color32::from_rgb(45, 45, 45);
const INPUT_BORDER: egui::Color32 = egui::Color32::from_rgb(70, 70, 70);
const INPUT_BORDER_FOCUS: egui::Color32 = egui::Color32::from_rgb(217, 119, 80);
const MEMORY_COLOR: egui::Color32 = egui::Color32::from_rgb(80, 65, 50);

// Context budget (conservative estimates in chars, ~4 chars/token)
const SUMMARY_BUDGET_CHARS: usize = 2000; // ~500 tokens for compressed memory
const RAW_WINDOW: usize = 2; // keep last N exchanges raw

#[derive(Clone, Copy, PartialEq)]
enum Role {
    User,
    Assistant,
}

struct ChatMessage {
    role: Role,
    text: String,
}

enum InferEvent {
    Token(String),
    Stderr(String),
    Done,
}

struct ChatApp {
    messages: Vec<ChatMessage>,
    input: String,
    model_path: String,
    binary_path: String,
    max_tokens: u32,
    temperature: f32,
    rx: Option<mpsc::Receiver<InferEvent>>,
    child: Option<Child>,
    generating: bool,
    stats_line: String,
    scroll_to_bottom: bool,
    // Recursive self-distillation
    summary: Arc<Mutex<String>>,
    compressing: Arc<Mutex<bool>>,
    turn_count: usize,
}

impl ChatApp {
    fn new(model_path: String, binary_path: String) -> Self {
        Self {
            messages: vec![],
            input: String::new(),
            model_path,
            binary_path,
            max_tokens: 256,
            temperature: 0.7,
            rx: None,
            child: None,
            generating: false,
            stats_line: String::new(),
            scroll_to_bottom: false,
            summary: Arc::new(Mutex::new(String::new())),
            compressing: Arc::new(Mutex::new(false)),
            turn_count: 0,
        }
    }

    /// Build prompt with compressed memory + raw recent window
    fn build_prompt(&self) -> String {
        let summary = self.summary.lock().unwrap().clone();

        // Count exchanges (user+assistant pairs)
        let mut exchanges: Vec<(usize, usize)> = vec![];
        let mut i = 0;
        while i < self.messages.len() {
            if self.messages[i].role == Role::User {
                let user_idx = i;
                let asst_idx = if i + 1 < self.messages.len() {
                    i + 1
                } else {
                    i
                };
                exchanges.push((user_idx, asst_idx));
                i += 2;
            } else {
                i += 1;
            }
        }

        let mut prompt = String::from("<s>");

        // If we have a summary and more exchanges than the raw window, inject it
        if !summary.is_empty() && exchanges.len() > RAW_WINDOW {
            prompt.push_str(&format!(
                "[INST] Context from earlier conversation:\n{}\n\nContinue the conversation. [/INST] Understood, I have the context.</s>",
                summary
            ));
        }

        // Only include the last RAW_WINDOW exchanges as raw text
        let start = if exchanges.len() > RAW_WINDOW {
            exchanges.len() - RAW_WINDOW
        } else {
            0
        };

        for &(user_idx, asst_idx) in &exchanges[start..] {
            prompt.push_str(&format!(
                "[INST] {} [/INST]",
                self.messages[user_idx].text
            ));
            if asst_idx != user_idx && !self.messages[asst_idx].text.is_empty() {
                prompt.push_str(&self.messages[asst_idx].text);
                // Don't close the last assistant message (it's generating)
                if asst_idx != self.messages.len() - 1 {
                    prompt.push_str("</s>");
                }
            }
        }

        prompt
    }

    /// Spawn background compression after a turn completes
    fn compress_turn(&mut self, user_text: String, assistant_text: String) {
        let summary = Arc::clone(&self.summary);
        let compressing = Arc::clone(&self.compressing);
        let binary = self.binary_path.clone();
        let model = self.model_path.clone();

        // Don't stack compressions
        if *compressing.lock().unwrap() {
            return;
        }
        *compressing.lock().unwrap() = true;

        std::thread::spawn(move || {
            let old_summary = summary.lock().unwrap().clone();

            let compress_prompt = if old_summary.is_empty() {
                format!(
                    "<s>[INST] Compress this conversation exchange into key facts, decisions, names, numbers, and context. Be concise, under 80 words. Output only the summary, nothing else.\n\nUser: {}\nAssistant: {} [/INST]",
                    user_text, assistant_text
                )
            } else {
                format!(
                    "<s>[INST] Update this conversation summary with the new exchange. Keep key facts, decisions, names, numbers. Stay under 80 words. Output only the updated summary.\n\nPrevious summary: {}\n\nNew exchange:\nUser: {}\nAssistant: {} [/INST]",
                    old_summary, user_text, assistant_text
                )
            };

            let output = Command::new(&binary)
                .args([
                    "--model",
                    &model,
                    "--prompt",
                    &compress_prompt,
                    "--tokens",
                    "120",
                    "--temp",
                    "0.3",
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .output();

            if let Ok(out) = output {
                let new_summary = String::from_utf8_lossy(&out.stdout)
                    .replace('\u{2581}', " ")
                    .trim()
                    .to_string();

                if !new_summary.is_empty() {
                    // Truncate if over budget
                    let truncated = if new_summary.len() > SUMMARY_BUDGET_CHARS {
                        new_summary[..SUMMARY_BUDGET_CHARS].to_string()
                    } else {
                        new_summary
                    };
                    *summary.lock().unwrap() = truncated;
                }
            }

            *compressing.lock().unwrap() = false;
        });
    }

    fn send_message(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.generating {
            return;
        }
        self.input.clear();
        self.messages.push(ChatMessage {
            role: Role::User,
            text,
        });
        self.messages.push(ChatMessage {
            role: Role::Assistant,
            text: String::new(),
        });

        let prompt = self.build_prompt();
        self.stats_line.clear();
        self.scroll_to_bottom = true;

        match Command::new(&self.binary_path)
            .args([
                "--model",
                &self.model_path,
                "--prompt",
                &prompt,
                "--tokens",
                &self.max_tokens.to_string(),
                "--temp",
                &format!("{:.2}", self.temperature),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(mut child) => {
                let (tx, rx) = mpsc::channel();
                let stdout = child.stdout.take().unwrap();
                let stderr = child.stderr.take().unwrap();

                let tx_out = tx.clone();
                std::thread::spawn(move || {
                    let mut reader = std::io::BufReader::with_capacity(64, stdout);
                    let mut buf = [0u8; 64];
                    let mut pending = Vec::new();
                    loop {
                        match reader.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => {
                                pending.extend_from_slice(&buf[..n]);
                                match std::str::from_utf8(&pending) {
                                    Ok(s) => {
                                        let _ = tx_out.send(InferEvent::Token(s.to_string()));
                                        pending.clear();
                                    }
                                    Err(e) => {
                                        let valid_up_to = e.valid_up_to();
                                        if valid_up_to > 0 {
                                            let s = std::str::from_utf8(&pending[..valid_up_to])
                                                .unwrap()
                                                .to_string();
                                            let _ = tx_out.send(InferEvent::Token(s));
                                            pending.drain(..valid_up_to);
                                        }
                                    }
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    if !pending.is_empty() {
                        let s = String::from_utf8_lossy(&pending).to_string();
                        let _ = tx_out.send(InferEvent::Token(s));
                    }
                    let _ = tx_out.send(InferEvent::Done);
                });

                std::thread::spawn(move || {
                    use std::io::BufRead;
                    let reader = std::io::BufReader::new(stderr);
                    for line in reader.lines().flatten() {
                        let _ = tx.send(InferEvent::Stderr(line));
                    }
                });

                self.child = Some(child);
                self.rx = Some(rx);
                self.generating = true;
            }
            Err(e) => {
                if let Some(msg) = self.messages.last_mut() {
                    msg.text = format!("Error: failed to spawn inference binary: {e}");
                }
            }
        }
    }

    fn stop_generation(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.child = None;
        self.generating = false;
    }

    fn drain_events(&mut self) {
        let Some(rx) = &self.rx else { return };
        let mut pending_compress: Option<(String, String)> = None;

        while let Ok(event) = rx.try_recv() {
            match event {
                InferEvent::Token(t) => {
                    if let Some(msg) = self.messages.last_mut() {
                        msg.text.push_str(&t.replace('\u{2581}', " "));
                    }
                    self.scroll_to_bottom = true;
                }
                InferEvent::Stderr(line) => {
                    if line.contains("tok/s") || line.contains("ms/token") {
                        self.stats_line = line;
                    }
                }
                InferEvent::Done => {
                    self.generating = false;
                    self.child = None;
                    self.turn_count += 1;

                    if self.turn_count > RAW_WINDOW {
                        let compress_idx = (self.turn_count - RAW_WINDOW - 1) * 2;
                        if compress_idx + 1 < self.messages.len() {
                            pending_compress = Some((
                                self.messages[compress_idx].text.clone(),
                                self.messages[compress_idx + 1].text.clone(),
                            ));
                        }
                    }
                }
            }
        }

        // Fire compression outside the rx borrow
        if let Some((user_text, asst_text)) = pending_compress {
            self.compress_turn(user_text, asst_text);
        }
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();

        if self.generating {
            ctx.request_repaint();
        }

        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = BG;
        visuals.window_fill = BG;
        visuals.extreme_bg_color = INPUT_BG;
        ctx.set_visuals(visuals);

        let is_compressing = *self.compressing.lock().unwrap();
        let has_summary = !self.summary.lock().unwrap().is_empty();

        // Top bar
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new("Mistral 7B")
                        .size(18.0)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
                ui.label(
                    egui::RichText::new("Q4_0 \u{00b7} local")
                        .size(13.0)
                        .color(TEXT_SECONDARY),
                );

                // Memory indicator
                if is_compressing {
                    ui.label(
                        egui::RichText::new("\u{2699} compressing...")
                            .size(11.0)
                            .color(MEMORY_COLOR),
                    );
                } else if has_summary {
                    ui.label(
                        egui::RichText::new(&format!("\u{1f9e0} {} turns memorized", self.turn_count.saturating_sub(RAW_WINDOW)))
                            .size(11.0)
                            .color(MEMORY_COLOR),
                    );
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(12.0);
                    if !self.stats_line.is_empty() {
                        ui.label(
                            egui::RichText::new(&self.stats_line)
                                .size(11.0)
                                .monospace()
                                .color(TEXT_SECONDARY),
                        );
                    }
                });
            });
            ui.add_space(6.0);
            ui.separator();
        });

        // Bottom input panel
        egui::TopBottomPanel::bottom("input_panel")
            .min_height(60.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.add_space(12.0);
                    let available_w = ui.available_width() - 80.0;

                    let input_frame = egui::Frame::NONE
                        .fill(INPUT_BG)
                        .corner_radius(egui::CornerRadius::same(12))
                        .stroke(egui::Stroke::new(
                            1.0,
                            if self.generating {
                                INPUT_BORDER
                            } else {
                                INPUT_BORDER_FOCUS
                            },
                        ))
                        .inner_margin(egui::Margin::symmetric(12, 8));

                    input_frame.show(ui, |ui| {
                        let resp = ui.add_sized(
                            [available_w, 24.0],
                            egui::TextEdit::singleline(&mut self.input)
                                .font(egui::TextStyle::Body)
                                .text_color(TEXT_PRIMARY)
                                .frame(false)
                                .hint_text(
                                    egui::RichText::new("Send a message...")
                                        .color(TEXT_SECONDARY),
                                ),
                        );

                        let enter =
                            resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        if enter && !self.input.is_empty() && !self.generating {
                            self.send_message();
                            resp.request_focus();
                        }
                    });

                    ui.add_space(4.0);

                    if self.generating {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("\u{25a0}").size(16.0).color(TEXT_PRIMARY),
                                )
                                .fill(ACCENT)
                                .corner_radius(egui::CornerRadius::same(8))
                                .min_size(egui::vec2(40.0, 36.0)),
                            )
                            .clicked()
                        {
                            self.stop_generation();
                        }
                    } else if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("\u{2191}").size(16.0).color(TEXT_PRIMARY),
                            )
                            .fill(if self.input.is_empty() {
                                INPUT_BORDER
                            } else {
                                ACCENT
                            })
                            .corner_radius(egui::CornerRadius::same(8))
                            .min_size(egui::vec2(40.0, 36.0)),
                        )
                        .clicked()
                    {
                        if !self.input.is_empty() {
                            self.send_message();
                        }
                    }
                    ui.add_space(4.0);
                });
                ui.add_space(8.0);
            });

        // Chat messages
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin::ZERO))
            .show(ctx, |ui| {
                let scroll = egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true);

                scroll.show(ui, |ui| {
                    ui.add_space(16.0);
                    let max_width = (ui.available_width() * 0.75).min(680.0);

                    if self.messages.is_empty() {
                        ui.vertical_centered(|ui| {
                            ui.add_space(100.0);
                            ui.label(
                                egui::RichText::new("Mistral 7B")
                                    .size(28.0)
                                    .strong()
                                    .color(TEXT_PRIMARY),
                            );
                            ui.add_space(8.0);
                            ui.label(
                                egui::RichText::new(
                                    "Running locally on Apple Silicon \u{00b7} Q4_0 quantized",
                                )
                                .size(14.0)
                                .color(TEXT_SECONDARY),
                            );
                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new(
                                    "Recursive self-distillation \u{00b7} infinite context",
                                )
                                .size(12.0)
                                .color(MEMORY_COLOR),
                            );
                        });
                    }

                    for msg in &self.messages {
                        if msg.role == Role::Assistant && msg.text.is_empty() && self.generating {
                            ui.horizontal(|ui| {
                                ui.add_space(20.0);
                                ui.label(
                                    egui::RichText::new("\u{25cf} \u{25cf} \u{25cf}")
                                        .size(14.0)
                                        .color(TEXT_SECONDARY),
                                );
                            });
                            continue;
                        }

                        match msg.role {
                            Role::User => {
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::TOP),
                                    |ui| {
                                        ui.add_space(20.0);
                                        let frame = egui::Frame::NONE
                                            .fill(USER_BUBBLE)
                                            .corner_radius(egui::CornerRadius {
                                                nw: 16,
                                                ne: 4,
                                                sw: 16,
                                                se: 16,
                                            })
                                            .inner_margin(egui::Margin::symmetric(14, 10));

                                        frame.show(ui, |ui| {
                                            ui.set_max_width(max_width);
                                            ui.label(
                                                egui::RichText::new(&msg.text)
                                                    .size(14.0)
                                                    .color(TEXT_PRIMARY),
                                            );
                                        });
                                    },
                                );
                            }
                            Role::Assistant => {
                                ui.indent("asst_indent", |ui| {
                                    let frame = egui::Frame::NONE
                                        .fill(ASSISTANT_BUBBLE)
                                        .corner_radius(egui::CornerRadius {
                                            nw: 4,
                                            ne: 16,
                                            sw: 16,
                                            se: 16,
                                        })
                                        .inner_margin(egui::Margin::symmetric(14, 10));

                                    frame.show(ui, |ui| {
                                        ui.set_max_width(max_width);
                                        ui.label(
                                            egui::RichText::new(&msg.text)
                                                .size(14.0)
                                                .color(TEXT_PRIMARY),
                                        );
                                    });
                                });
                            }
                        }
                        ui.add_space(8.0);
                    }
                    ui.add_space(16.0);
                });
            });
    }
}

fn main() -> eframe::Result<()> {
    let model_path = std::env::args()
        .skip_while(|a| a != "--model")
        .nth(1)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap()
                .join("models/mistral-7b-instruct-v0.2.Q4_0.gguf")
                .to_string_lossy()
                .to_string()
        });

    let binary_path = std::env::args()
        .skip_while(|a| a != "--binary")
        .nth(1)
        .unwrap_or_else(|| {
            let cwd = std::env::current_dir().unwrap();
            let candidate = cwd.join("mistral/mistral");
            if candidate.exists() {
                return candidate.to_string_lossy().to_string();
            }
            if let Ok(exe) = std::env::current_exe() {
                let candidate = exe
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join("mistral/mistral");
                if candidate.exists() {
                    return candidate.to_string_lossy().to_string();
                }
            }
            "mistral/mistral".to_string()
        });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([760.0, 600.0])
            .with_title("Mistral Chat")
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Mistral Chat",
        options,
        Box::new(move |_cc| Ok(Box::new(ChatApp::new(model_path, binary_path)))),
    )
}

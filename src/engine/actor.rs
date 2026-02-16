//! Actor-based concurrency for inference.
//!
//! The actor model isolates the inference engine in a dedicated OS thread,
//! preventing the blocking `mx::eval()` calls from freezing the UI thread.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     mpsc channel      ┌──────────────────┐
//! │  UI Thread   │ ──── Command ────────>│ Inference Actor   │
//! │  (Tauri)     │ <──── Event ──────────│ (Dedicated Thread)│
//! │              │     mpsc channel      │  Owns: Model,     │
//! │              │                       │  Tokenizer, Cache  │
//! └─────────────┘                       └──────────────────┘
//! ```
//!
//! # Token Batching
//!
//! M3/M4 chips can generate 100+ tokens/second. Emitting individual events
//! for each token saturates the IPC layer. The actor buffers tokens and
//! emits chunks every ~50ms or every 10 tokens.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::engine::InferenceEngine;
use crate::error::Result;
use crate::sampling::SamplingConfig;

/// Maximum tokens to buffer before emitting to the UI.
const TOKEN_BATCH_SIZE: usize = 10;

/// Maximum time to buffer tokens before emitting (milliseconds).
const TOKEN_BATCH_INTERVAL_MS: u64 = 50;

/// Commands sent from the UI thread to the inference actor.
pub enum ActorCommand {
    /// Start generation with tokenized prompt.
    Generate {
        prompt_tokens: Vec<i32>,
        sampling_config: SamplingConfig,
        max_tokens: usize,
        stop_tokens: Vec<i32>,
    },

    /// Stop the current generation.
    Cancel,

    /// Reset the engine for a new conversation.
    Reset,

    /// Shut down the actor thread.
    Shutdown,
}

/// Events sent from the inference actor back to the UI thread.
#[derive(Debug, Clone)]
pub enum ActorEvent {
    /// A batch of generated tokens (for streaming display).
    TokenBatch(Vec<i32>),

    /// Generation completed successfully.
    Done {
        generated_tokens: Vec<i32>,
        prompt_tokens: usize,
        tokens_per_second: f64,
    },

    /// An error occurred.
    Error(String),

    /// The actor has shut down.
    Stopped,
}

/// Handle to the inference actor, used by the UI thread.
pub struct ActorHandle {
    /// Send commands to the actor.
    cmd_tx: Sender<ActorCommand>,

    /// Receive events from the actor.
    event_rx: Receiver<ActorEvent>,

    /// Shared cancellation flag.
    is_running: Arc<AtomicBool>,

    /// Handle to the actor thread.
    thread: Option<thread::JoinHandle<()>>,
}

impl ActorHandle {
    /// Spawn a new inference actor on a dedicated OS thread.
    pub fn spawn(engine: InferenceEngine) -> Self {
        let (cmd_tx, cmd_rx) = bounded::<ActorCommand>(16);
        let (event_tx, event_rx) = bounded::<ActorEvent>(256);
        let is_running = engine.running_flag();

        let thread = thread::Builder::new()
            .name("inference-actor".into())
            .spawn(move || {
                actor_loop(engine, cmd_rx, event_tx);
            })
            .expect("failed to spawn inference actor thread");

        ActorHandle {
            cmd_tx,
            event_rx,
            is_running,
            thread: Some(thread),
        }
    }

    /// Send a generation request.
    pub fn generate(
        &self,
        prompt_tokens: Vec<i32>,
        sampling_config: SamplingConfig,
        max_tokens: usize,
        stop_tokens: Vec<i32>,
    ) -> Result<()> {
        self.cmd_tx
            .send(ActorCommand::Generate {
                prompt_tokens,
                sampling_config,
                max_tokens,
                stop_tokens,
            })
            .map_err(|_| crate::error::LlamaError::Inference("actor channel closed".into()))
    }

    /// Cancel the current generation.
    pub fn cancel(&self) {
        self.is_running.store(false, Ordering::Release);
        let _ = self.cmd_tx.send(ActorCommand::Cancel);
    }

    /// Reset the engine state.
    pub fn reset(&self) -> Result<()> {
        self.cmd_tx
            .send(ActorCommand::Reset)
            .map_err(|_| crate::error::LlamaError::Inference("actor channel closed".into()))
    }

    /// Shut down the actor thread.
    pub fn shutdown(mut self) {
        let _ = self.cmd_tx.send(ActorCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    /// Try to receive the next event (non-blocking).
    pub fn try_recv(&self) -> Option<ActorEvent> {
        self.event_rx.try_recv().ok()
    }

    /// Receive the next event (blocking).
    pub fn recv(&self) -> Option<ActorEvent> {
        self.event_rx.recv().ok()
    }

    /// Receive with timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<ActorEvent> {
        self.event_rx.recv_timeout(timeout).ok()
    }

    /// Whether generation is currently running.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }
}

impl Drop for ActorHandle {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(ActorCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The main loop running on the dedicated inference thread.
fn actor_loop(
    mut engine: InferenceEngine,
    cmd_rx: Receiver<ActorCommand>,
    event_tx: Sender<ActorEvent>,
) {
    loop {
        match cmd_rx.recv() {
            Ok(ActorCommand::Generate {
                prompt_tokens,
                sampling_config,
                max_tokens,
                stop_tokens,
            }) => {
                run_generation(
                    &mut engine,
                    &prompt_tokens,
                    &sampling_config,
                    max_tokens,
                    &stop_tokens,
                    &event_tx,
                );
            }

            Ok(ActorCommand::Cancel) => {
                engine.cancel();
            }

            Ok(ActorCommand::Reset) => {
                engine.reset();
            }

            Ok(ActorCommand::Shutdown) => {
                let _ = event_tx.send(ActorEvent::Stopped);
                break;
            }

            Err(_) => {
                // Channel closed, exit
                break;
            }
        }
    }
}

/// Execute a generation request with token batching.
fn run_generation(
    engine: &mut InferenceEngine,
    prompt_tokens: &[i32],
    sampling_config: &SamplingConfig,
    max_tokens: usize,
    stop_tokens: &[i32],
    event_tx: &Sender<ActorEvent>,
) {
    let mut all_tokens = Vec::new();
    let mut batch_buffer = Vec::with_capacity(TOKEN_BATCH_SIZE);
    let mut last_emit = Instant::now();

    let result = engine.generate(
        prompt_tokens,
        sampling_config,
        max_tokens,
        stop_tokens,
        |token_id| {
            all_tokens.push(token_id);
            batch_buffer.push(token_id);

            // Emit batch if buffer is full or interval elapsed
            let should_emit = batch_buffer.len() >= TOKEN_BATCH_SIZE
                || last_emit.elapsed() >= Duration::from_millis(TOKEN_BATCH_INTERVAL_MS);

            if should_emit && !batch_buffer.is_empty() {
                let _ = event_tx.send(ActorEvent::TokenBatch(batch_buffer.clone()));
                batch_buffer.clear();
                last_emit = Instant::now();
            }
        },
    );

    // Flush remaining tokens
    if !batch_buffer.is_empty() {
        let _ = event_tx.send(ActorEvent::TokenBatch(batch_buffer));
    }

    match result {
        Ok((tokens, stats)) => {
            let _ = event_tx.send(ActorEvent::Done {
                generated_tokens: tokens,
                prompt_tokens: stats.prompt_tokens,
                tokens_per_second: stats.tokens_per_second,
            });
        }
        Err(e) => {
            let _ = event_tx.send(ActorEvent::Error(e.to_string()));
        }
    }
}

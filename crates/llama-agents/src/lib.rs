//! # llama-agents
//!
//! Agent orchestration primitives for llama.rs and oxidizedgraph integration.
//!
//! This crate provides:
//! - Async agent nodes (`Generate` and `Summarize`) implemented against `LlamaEngine`
//! - Actor-style state persistence for multi-session chat orchestration
//! - Multi-turn history + persisted `llama-kv` state per session

use llama_engine::{LlamaEngine, Session};
use llama_kv::{KVLayout, SessionKVCache};
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("engine error: {0}")]
    Engine(#[from] llama_engine::LlamaError),
    #[error("kv error: {0}")]
    Kv(#[from] llama_kv::KVError),
    #[error("state store unavailable")]
    StoreUnavailable,
    #[error("session {0} not found")]
    SessionNotFound(Uuid),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct PersistedSession {
    pub session_id: Uuid,
    pub history: Vec<ChatTurn>,
    pub kv_cache: SessionKVCache,
}

impl PersistedSession {
    pub fn new(session_id: Uuid, kv_cache: SessionKVCache) -> Self {
        Self {
            session_id,
            history: Vec::new(),
            kv_cache,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CompletionPolicy {
    Free,
    Archive,
}

#[derive(Debug, Clone, Copy)]
pub struct AgentConfig {
    pub n_layers: usize,
    pub max_seq_len: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub layout: KVLayout,
    pub default_generate_tokens: usize,
    pub summary_tokens: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            n_layers: 2,
            max_seq_len: 64,
            n_heads: 2,
            head_dim: 4,
            layout: KVLayout::BySequence,
            default_generate_tokens: 16,
            summary_tokens: 8,
        }
    }
}

impl AgentConfig {
    fn new_cache(&self) -> SessionKVCache {
        SessionKVCache::new(
            self.n_layers,
            self.max_seq_len,
            self.n_heads,
            self.head_dim,
            self.layout,
        )
    }
}

enum Command {
    Upsert(
        PersistedSession,
        mpsc::Sender<std::result::Result<(), AgentError>>,
    ),
    Get(
        Uuid,
        mpsc::Sender<std::result::Result<Option<PersistedSession>, AgentError>>,
    ),
    Complete(
        Uuid,
        CompletionPolicy,
        mpsc::Sender<std::result::Result<(), AgentError>>,
    ),
    ActiveCount(mpsc::Sender<usize>),
    ArchivedCount(mpsc::Sender<usize>),
    ActiveBytes(mpsc::Sender<usize>),
    PeakBytes(mpsc::Sender<usize>),
    Shutdown,
}

#[derive(Debug, Default)]
struct ActorState {
    active: HashMap<Uuid, PersistedSession>,
    archived: HashMap<Uuid, PersistedSession>,
    peak_active_bytes: usize,
}

impl ActorState {
    fn active_bytes(&self) -> usize {
        self.active
            .values()
            .map(|s| s.kv_cache.memory_bytes())
            .sum()
    }

    fn update_peak(&mut self) {
        self.peak_active_bytes = self.peak_active_bytes.max(self.active_bytes());
    }
}

#[derive(Debug)]
pub struct AgentStateStore {
    tx: mpsc::Sender<Command>,
    handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
}

impl Clone for AgentStateStore {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            handle: Arc::clone(&self.handle),
        }
    }
}

impl AgentStateStore {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Command>();
        let handle = thread::spawn(move || {
            let mut state = ActorState::default();
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    Command::Upsert(session, reply) => {
                        state.active.insert(session.session_id, session);
                        state.update_peak();
                        let _ = reply.send(Ok(()));
                    }
                    Command::Get(id, reply) => {
                        let session = state
                            .active
                            .get(&id)
                            .cloned()
                            .or_else(|| state.archived.get(&id).cloned());
                        let _ = reply.send(Ok(session));
                    }
                    Command::Complete(id, policy, reply) => {
                        let maybe = state.active.remove(&id);
                        let result = if let Some(session) = maybe {
                            if matches!(policy, CompletionPolicy::Archive) {
                                state.archived.insert(id, session);
                            }
                            Ok(())
                        } else {
                            Err(AgentError::SessionNotFound(id))
                        };
                        let _ = reply.send(result);
                    }
                    Command::ActiveCount(reply) => {
                        let _ = reply.send(state.active.len());
                    }
                    Command::ArchivedCount(reply) => {
                        let _ = reply.send(state.archived.len());
                    }
                    Command::ActiveBytes(reply) => {
                        let _ = reply.send(state.active_bytes());
                    }
                    Command::PeakBytes(reply) => {
                        let _ = reply.send(state.peak_active_bytes);
                    }
                    Command::Shutdown => break,
                }
            }
        });
        Self {
            tx,
            handle: Arc::new(Mutex::new(Some(handle))),
        }
    }

    pub fn upsert(&self, session: PersistedSession) -> std::result::Result<(), AgentError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Upsert(session, reply_tx))
            .map_err(|_| AgentError::StoreUnavailable)?;
        reply_rx.recv().map_err(|_| AgentError::StoreUnavailable)?
    }

    pub fn get(
        &self,
        session_id: Uuid,
    ) -> std::result::Result<Option<PersistedSession>, AgentError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Get(session_id, reply_tx))
            .map_err(|_| AgentError::StoreUnavailable)?;
        reply_rx.recv().map_err(|_| AgentError::StoreUnavailable)?
    }

    pub fn complete(
        &self,
        session_id: Uuid,
        policy: CompletionPolicy,
    ) -> std::result::Result<(), AgentError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Complete(session_id, policy, reply_tx))
            .map_err(|_| AgentError::StoreUnavailable)?;
        reply_rx.recv().map_err(|_| AgentError::StoreUnavailable)?
    }

    pub fn active_count(&self) -> usize {
        let (reply_tx, reply_rx) = mpsc::channel();
        let _ = self.tx.send(Command::ActiveCount(reply_tx));
        reply_rx.recv().unwrap_or(0)
    }

    pub fn archived_count(&self) -> usize {
        let (reply_tx, reply_rx) = mpsc::channel();
        let _ = self.tx.send(Command::ArchivedCount(reply_tx));
        reply_rx.recv().unwrap_or(0)
    }

    pub fn active_memory_bytes(&self) -> usize {
        let (reply_tx, reply_rx) = mpsc::channel();
        let _ = self.tx.send(Command::ActiveBytes(reply_tx));
        reply_rx.recv().unwrap_or(0)
    }

    pub fn peak_active_memory_bytes(&self) -> usize {
        let (reply_tx, reply_rx) = mpsc::channel();
        let _ = self.tx.send(Command::PeakBytes(reply_tx));
        reply_rx.recv().unwrap_or(0)
    }
}

impl Default for AgentStateStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AgentStateStore {
    fn drop(&mut self) {
        if Arc::strong_count(&self.handle) == 1 {
            let _ = self.tx.send(Command::Shutdown);
            if let Some(handle) = self.handle.lock().ok().and_then(|mut g| g.take()) {
                let _ = handle.join();
            }
        }
    }
}

fn advance_kv_state(
    cache: &mut SessionKVCache,
    token_count: usize,
) -> std::result::Result<(), AgentError> {
    for _ in 0..token_count {
        let mut k_owned = Vec::with_capacity(cache.n_layers());
        let mut v_owned = Vec::with_capacity(cache.n_layers());
        for i in 0..cache.n_layers() {
            let layer = cache
                .layer(i)
                .expect("layer index should be valid during kv state update");
            let width = layer.n_heads * layer.head_dim;
            k_owned.push(vec![0.0f32; width]);
            v_owned.push(vec![0.0f32; width]);
        }
        let k_refs: Vec<&[f32]> = k_owned.iter().map(Vec::as_slice).collect();
        let v_refs: Vec<&[f32]> = v_owned.iter().map(Vec::as_slice).collect();
        cache.append_token(&k_refs, &v_refs)?;
    }
    Ok(())
}

pub struct LlamaAgentNodes<'a, E: LlamaEngine> {
    engine: &'a E,
    store: &'a AgentStateStore,
    config: AgentConfig,
}

impl<'a, E: LlamaEngine> LlamaAgentNodes<'a, E> {
    pub fn new(engine: &'a E, store: &'a AgentStateStore, config: AgentConfig) -> Self {
        Self {
            engine,
            store,
            config,
        }
    }

    pub async fn generate(
        &self,
        session: &mut Session,
        prompt: &str,
        max_tokens: usize,
    ) -> std::result::Result<String, AgentError> {
        let prompt_tokens = self.engine.tokenize(prompt)?;
        let _ = self.engine.prefill(session, &prompt_tokens)?;

        let mut generated = Vec::new();
        for _ in 0..max_tokens {
            let step = self.engine.decode(session)?;
            generated.push(step.token);
        }
        let output = self.engine.detokenize(&generated)?;

        let session_id = session.id();
        let mut persisted = self
            .store
            .get(session_id)?
            .unwrap_or_else(|| PersistedSession::new(session_id, self.config.new_cache()));
        persisted.history.push(ChatTurn {
            role: ChatRole::User,
            content: prompt.to_string(),
        });
        persisted.history.push(ChatTurn {
            role: ChatRole::Assistant,
            content: output.clone(),
        });
        advance_kv_state(
            &mut persisted.kv_cache,
            prompt_tokens.len() + generated.len(),
        )?;
        self.store.upsert(persisted)?;
        Ok(output)
    }

    pub async fn summarize(
        &self,
        session: &mut Session,
        text: &str,
    ) -> std::result::Result<String, AgentError> {
        let prompt = format!("Summarize: {text}");
        self.generate(session, &prompt, self.config.summary_tokens)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use llama_engine::{DecodeResult, LlamaError, ModelHandle, ModelSpec, PrefillResult, TokenId};
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::Barrier;

    #[derive(Default)]
    struct TestEngine {
        next_token: AtomicI32,
    }

    impl LlamaEngine for TestEngine {
        fn load_model(&self, _spec: &ModelSpec) -> std::result::Result<ModelHandle, LlamaError> {
            Ok(ModelHandle)
        }

        fn tokenize(&self, text: &str) -> std::result::Result<Vec<TokenId>, LlamaError> {
            let n = text.split_whitespace().count();
            Ok((0..n as i32).collect())
        }

        fn detokenize(&self, tokens: &[TokenId]) -> std::result::Result<String, LlamaError> {
            Ok(tokens
                .iter()
                .map(|t| format!("tok{t}"))
                .collect::<Vec<_>>()
                .join(" "))
        }

        fn prefill(
            &self,
            _session: &mut Session,
            tokens: &[TokenId],
        ) -> std::result::Result<PrefillResult, LlamaError> {
            Ok(PrefillResult {
                tokens_processed: tokens.len(),
            })
        }

        fn decode(&self, _session: &mut Session) -> std::result::Result<DecodeResult, LlamaError> {
            let token = self.next_token.fetch_add(1, Ordering::SeqCst);
            Ok(DecodeResult { token })
        }

        fn embed(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, LlamaError> {
            Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
        }
    }

    #[test]
    fn generate_node_is_async_and_persists_history_and_kv() {
        let engine = TestEngine::default();
        let store = AgentStateStore::new();
        let nodes = LlamaAgentNodes::new(&engine, &store, AgentConfig::default());
        let mut session = Session::new();

        let out = block_on(nodes.generate(&mut session, "hello world", 3)).unwrap();
        assert!(out.contains("tok"));

        let persisted = store.get(session.id()).unwrap().unwrap();
        assert_eq!(persisted.history.len(), 2);
        assert_eq!(persisted.history[0].role, ChatRole::User);
        assert_eq!(persisted.history[1].role, ChatRole::Assistant);
        assert_eq!(persisted.kv_cache.seq_len(), 5); // 2 prompt + 3 generated
    }

    #[test]
    fn summarize_node_is_async_and_updates_session_state() {
        let engine = TestEngine::default();
        let store = AgentStateStore::new();
        let cfg = AgentConfig {
            summary_tokens: 2,
            ..AgentConfig::default()
        };
        let nodes = LlamaAgentNodes::new(&engine, &store, cfg);
        let mut session = Session::new();

        let summary = block_on(nodes.summarize(&mut session, "a long message")).unwrap();
        assert!(summary.contains("tok"));

        let persisted = store.get(session.id()).unwrap().unwrap();
        assert!(persisted.history[0].content.starts_with("Summarize:"));
        assert_eq!(persisted.history.len(), 2);
    }

    #[test]
    fn session_completion_frees_or_archives_kv_by_policy() {
        let store = AgentStateStore::new();
        let id = Uuid::new_v4();
        let session =
            PersistedSession::new(id, SessionKVCache::new(2, 32, 2, 4, KVLayout::BySequence));
        store.upsert(session).unwrap();
        assert_eq!(store.active_count(), 1);

        store.complete(id, CompletionPolicy::Archive).unwrap();
        assert_eq!(store.active_count(), 0);
        assert_eq!(store.archived_count(), 1);

        // Reinsert and free this time.
        let session2 =
            PersistedSession::new(id, SessionKVCache::new(2, 32, 2, 4, KVLayout::BySequence));
        store.upsert(session2).unwrap();
        store.complete(id, CompletionPolicy::Free).unwrap();
        assert_eq!(store.active_count(), 0);
        assert_eq!(store.archived_count(), 1);
    }

    #[test]
    fn memory_footprint_stays_bounded_under_high_concurrency() {
        let store = Arc::new(AgentStateStore::new());
        let sessions = 32usize;
        let rounds = 4usize;
        let per_session_bytes =
            SessionKVCache::new(2, 32, 2, 4, KVLayout::BySequence).memory_bytes();

        for _ in 0..rounds {
            let barrier = Arc::new(Barrier::new(sessions));
            let mut handles = Vec::with_capacity(sessions);

            for _ in 0..sessions {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                handles.push(thread::spawn(move || {
                    let sid = Uuid::new_v4();
                    let session = PersistedSession::new(
                        sid,
                        SessionKVCache::new(2, 32, 2, 4, KVLayout::BySequence),
                    );
                    store.upsert(session).unwrap();
                    barrier.wait();
                    store.complete(sid, CompletionPolicy::Free).unwrap();
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
            assert_eq!(store.active_count(), 0);
            assert_eq!(store.active_memory_bytes(), 0);
        }

        assert!(
            store.peak_active_memory_bytes() <= sessions * per_session_bytes,
            "peak bytes exceeded bound: peak={}, bound={}",
            store.peak_active_memory_bytes(),
            sessions * per_session_bytes
        );
    }
}

//! Session lifecycle management for inference requests.
//!
//! Tracks active sessions, enforces concurrency limits, and ensures
//! KV cache resources are freed when sessions complete or clients disconnect.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore, SemaphorePermit};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Tracks active inference sessions and controls concurrency.
pub struct SessionManager {
    /// Active sessions keyed by session ID.
    active: Mutex<HashMap<Uuid, SessionEntry>>,
    /// Semaphore limiting concurrent inference requests.
    concurrency_limit: Arc<Semaphore>,
    /// Maximum allowed concurrent sessions.
    max_concurrent: usize,
}

/// Metadata for an active session.
struct SessionEntry {
    /// Cancellation token to signal this session should stop.
    /// Stored here so external callers could cancel sessions by ID if needed.
    #[allow(dead_code)]
    cancel: CancellationToken,
}

/// A guard that releases session resources when dropped.
///
/// When the HTTP connection drops (client disconnect), the handler's
/// future is cancelled, this guard is dropped, and the session is
/// cleaned up automatically.
pub struct SessionGuard {
    session_id: Uuid,
    cancel: CancellationToken,
    manager: Arc<SessionManager>,
    _permit: SemaphorePermit<'static>,
}

impl SessionGuard {
    /// Get the session ID.
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Get a clone of the cancellation token to check in decode loops.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        // Signal cancellation so any in-flight decode loop stops.
        self.cancel.cancel();
        // Remove from active sessions (fire-and-forget).
        let manager = self.manager.clone();
        let id = self.session_id;
        tokio::spawn(async move {
            manager.remove_session(id).await;
        });
    }
}

impl SessionManager {
    /// Create a new session manager with the given concurrency limit.
    pub fn new(max_concurrent: usize) -> Arc<Self> {
        Arc::new(Self {
            active: Mutex::new(HashMap::new()),
            concurrency_limit: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        })
    }

    /// Acquire a session slot. Blocks if at concurrency limit.
    ///
    /// Returns a `SessionGuard` that automatically cleans up on drop.
    pub async fn acquire(self: &Arc<Self>, session_id: Uuid) -> SessionGuard {
        // SAFETY: We leak the Arc<Semaphore> reference to get a 'static permit.
        // The Semaphore lives as long as the SessionManager (Arc), which outlives all guards.
        let permit = unsafe {
            let semaphore: &'static Semaphore = &*Arc::as_ptr(&self.concurrency_limit);
            semaphore.acquire().await.expect("semaphore not closed")
        };

        let cancel = CancellationToken::new();

        {
            let mut active = self.active.lock().await;
            active.insert(
                session_id,
                SessionEntry {
                    cancel: cancel.clone(),
                },
            );
        }

        SessionGuard {
            session_id,
            cancel,
            manager: Arc::clone(self),
            _permit: permit,
        }
    }

    /// Try to acquire without blocking. Returns None if at capacity.
    pub async fn try_acquire(self: &Arc<Self>, session_id: Uuid) -> Option<SessionGuard> {
        // Same static lifetime trick as acquire().
        let permit = unsafe {
            let semaphore: &'static Semaphore = &*Arc::as_ptr(&self.concurrency_limit);
            semaphore.try_acquire().ok()?
        };

        let cancel = CancellationToken::new();

        {
            let mut active = self.active.lock().await;
            active.insert(
                session_id,
                SessionEntry {
                    cancel: cancel.clone(),
                },
            );
        }

        Some(SessionGuard {
            session_id,
            cancel,
            manager: Arc::clone(self),
            _permit: permit,
        })
    }

    /// Remove a session from the active set (called by SessionGuard::drop).
    async fn remove_session(&self, id: Uuid) {
        let mut active = self.active.lock().await;
        active.remove(&id);
    }

    /// Number of currently active sessions.
    pub async fn active_count(&self) -> usize {
        let active = self.active.lock().await;
        active.len()
    }

    /// Maximum concurrent sessions allowed.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Number of available slots.
    pub fn available_permits(&self) -> usize {
        self.concurrency_limit.available_permits()
    }
}

//! Retrieval-augmented prompt builder with citation support.
//!
//! Takes search results from oxidizedRAG and builds a grounded prompt
//! with inline citations that the model can reference in its response.

/// A citation referencing a source document used in the prompt.
#[derive(Debug, Clone)]
pub struct Citation {
    /// Citation index (e.g., [1], [2]).
    pub index: usize,
    /// Source identifier.
    pub source_id: String,
    /// Relevance score from retrieval.
    pub score: f32,
    /// Short snippet of the cited content.
    pub snippet: String,
}

/// A retrieved context chunk to include in the prompt.
#[derive(Debug, Clone)]
pub struct RetrievedContext {
    /// Content text from the retrieved document.
    pub content: String,
    /// Source identifier for citation.
    pub source_id: String,
    /// Relevance score (higher is better).
    pub score: f32,
}

/// Builds retrieval-augmented prompts with inline citations.
///
/// The prompt format:
/// ```text
/// Context:
/// [1] <content from source 1>
/// [2] <content from source 2>
///
/// Based on the context above, answer the following question.
/// Cite sources using [N] notation.
///
/// Question: <user query>
/// ```
pub struct RagPromptBuilder {
    contexts: Vec<RetrievedContext>,
    max_context_tokens: usize,
}

impl RagPromptBuilder {
    /// Create a new prompt builder.
    pub fn new() -> Self {
        Self {
            contexts: Vec::new(),
            max_context_tokens: 2048,
        }
    }

    /// Set maximum approximate context length (in characters, as a proxy for tokens).
    pub fn max_context_chars(mut self, max: usize) -> Self {
        self.max_context_tokens = max;
        self
    }

    /// Add a retrieved context chunk.
    pub fn add_context(&mut self, ctx: RetrievedContext) {
        self.contexts.push(ctx);
    }

    /// Build the augmented prompt with citations.
    ///
    /// Returns `(prompt, citations)` where prompt is the full text to send to the model
    /// and citations are the source references included.
    pub fn build(&self, query: &str) -> (String, Vec<Citation>) {
        let mut prompt = String::from("Context:\n");
        let mut citations = Vec::new();
        let mut total_chars = 0;

        // Sort by score descending — best results first
        let mut sorted: Vec<_> = self.contexts.iter().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, ctx) in sorted.iter().enumerate() {
            let idx = i + 1;
            let line = format!("[{}] {}\n", idx, ctx.content);

            // Approximate token budget check
            if total_chars + line.len() > self.max_context_tokens {
                break;
            }

            total_chars += line.len();
            prompt.push_str(&line);

            citations.push(Citation {
                index: idx,
                source_id: ctx.source_id.clone(),
                score: ctx.score,
                snippet: if ctx.content.len() > 100 {
                    format!("{}...", &ctx.content[..100])
                } else {
                    ctx.content.clone()
                },
            });
        }

        prompt.push_str(
            "\nBased on the context above, answer the following question. \
             Cite sources using [N] notation.\n\n",
        );
        prompt.push_str(&format!("Question: {}\n", query));

        (prompt, citations)
    }
}

impl Default for RagPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_prompt_with_citations() {
        let mut builder = RagPromptBuilder::new();
        builder.add_context(RetrievedContext {
            content: "Rust is a systems programming language.".to_string(),
            source_id: "doc-1".to_string(),
            score: 0.95,
        });
        builder.add_context(RetrievedContext {
            content: "Rust was first released in 2015.".to_string(),
            source_id: "doc-2".to_string(),
            score: 0.80,
        });

        let (prompt, citations) = builder.build("What is Rust?");

        assert!(prompt.contains("[1]"));
        assert!(prompt.contains("[2]"));
        assert!(prompt.contains("Rust is a systems programming language."));
        assert!(prompt.contains("Question: What is Rust?"));
        assert!(prompt.contains("Cite sources using [N] notation"));
        assert_eq!(citations.len(), 2);
        // Highest score first
        assert_eq!(citations[0].source_id, "doc-1");
        assert_eq!(citations[1].source_id, "doc-2");
    }

    #[test]
    fn respects_context_budget() {
        let mut builder = RagPromptBuilder::new().max_context_chars(50);
        builder.add_context(RetrievedContext {
            content: "A".repeat(40),
            source_id: "a".to_string(),
            score: 0.9,
        });
        builder.add_context(RetrievedContext {
            content: "B".repeat(40),
            source_id: "b".to_string(),
            score: 0.8,
        });

        let (_, citations) = builder.build("test");
        // Only first fits within budget
        assert_eq!(citations.len(), 1);
    }

    #[test]
    fn empty_context_still_builds() {
        let builder = RagPromptBuilder::new();
        let (prompt, citations) = builder.build("What is Rust?");
        assert!(prompt.contains("Question: What is Rust?"));
        assert!(citations.is_empty());
    }
}

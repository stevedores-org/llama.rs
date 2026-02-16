//! Tests for tokenizer and chat template formatting.

use llama::tokenizer::{special_tokens, ChatMessage};

#[test]
fn test_chat_message_construction() {
    let msg = ChatMessage::user("Hello");
    assert_eq!(msg.role, "user");
    assert_eq!(msg.content, "Hello");

    let msg = ChatMessage::system("Be helpful");
    assert_eq!(msg.role, "system");

    let msg = ChatMessage::assistant("Hi there");
    assert_eq!(msg.role, "assistant");
}

#[test]
fn test_special_tokens() {
    assert_eq!(special_tokens::BOS, "<|begin_of_text|>");
    assert_eq!(special_tokens::EOS, "<|end_of_text|>");
    assert_eq!(special_tokens::EOT, "<|eot_id|>");
    assert_eq!(special_tokens::BOS_ID, 128000);
    assert_eq!(special_tokens::EOS_ID, 128001);
    assert_eq!(special_tokens::EOT_ID, 128009);
}

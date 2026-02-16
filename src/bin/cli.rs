//! Command-line interface for llama-rs.
//!
//! Provides a simple interactive chat interface and model inspection utilities.

use std::io::{self, BufRead, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("info") => cmd_info(&args[2..]),
        Some("inspect") => cmd_inspect(&args[2..]),
        Some("chat") => cmd_chat(&args[2..]),
        Some("--help") | Some("-h") => print_help(),
        Some("--version") | Some("-V") => print_version(),
        None => print_help(),
        Some(cmd) => {
            eprintln!("Unknown command: {cmd}");
            eprintln!("Run 'llama-cli --help' for usage.");
            std::process::exit(1);
        }
    }
}

fn print_version() {
    println!("llama-rs v{}", llama::VERSION);
}

fn print_help() {
    print_version();
    println!("High-performance LLM inference on Apple Silicon using Rust and MLX");
    println!();
    println!("USAGE:");
    println!("    llama-cli <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("    info     Show model configuration presets");
    println!("    inspect  Inspect a safetensors model directory");
    println!("    chat     Start an interactive chat session");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help       Print help");
    println!("    -V, --version    Print version");
}

/// Show model configuration presets and architecture details.
fn cmd_info(args: &[String]) {
    let model_name = args.first().map(|s| s.as_str()).unwrap_or("8b");

    let config = match model_name {
        "8b" | "8B" => llama::ModelConfig::llama3_8b(),
        "70b" | "70B" => llama::ModelConfig::llama3_70b(),
        other => {
            eprintln!("Unknown model preset: {other}");
            eprintln!("Available: 8b, 70b");
            std::process::exit(1);
        }
    };

    println!("Llama 3 {} Configuration", model_name.to_uppercase());
    println!("{}", "=".repeat(50));
    println!("Model type:          {}", config.model_type);
    println!("Vocabulary size:     {}", config.vocab_size);
    println!("Hidden size:         {}", config.hidden_size);
    println!("Intermediate size:   {}", config.intermediate_size);
    println!("Num layers:          {}", config.num_hidden_layers);
    println!("Num attention heads: {}", config.num_attention_heads);
    println!("Num KV heads:        {}", config.num_key_value_heads);
    println!("Head dimension:      {}", config.head_dim);
    println!("GQA ratio:           {}:1", config.num_queries_per_kv());
    println!("RoPE theta:          {}", config.rope_theta);
    println!("Max context:         {}", config.max_position_embeddings);
    println!("RMSNorm eps:         {}", config.rms_norm_eps);
    println!();

    let params = config.estimated_params();
    let memory_fp16 = params * 2;
    let memory_4bit = params / 2;
    println!("Estimated parameters:  {:.1}B", params as f64 / 1e9);
    println!("Memory (FP16):         {:.1} GB", memory_fp16 as f64 / 1e9);
    println!("Memory (4-bit quant):  {:.1} GB", memory_4bit as f64 / 1e9);
}

/// Inspect a model directory for safetensors files and metadata.
fn cmd_inspect(args: &[String]) {
    let model_dir = match args.first() {
        Some(dir) => dir,
        None => {
            eprintln!("Usage: llama-cli inspect <MODEL_DIR>");
            std::process::exit(1);
        }
    };

    let path = std::path::Path::new(model_dir);

    // Check for config.json
    let config_path = path.join("config.json");
    if config_path.exists() {
        match llama::weights::load_config(&config_path) {
            Ok(config) => {
                println!("Model Configuration:");
                println!("  Type:        {}", config.model_type);
                println!("  Layers:      {}", config.num_hidden_layers);
                println!("  Hidden:      {}", config.hidden_size);
                println!(
                    "  Heads:       {}/{} (Q/KV)",
                    config.num_attention_heads, config.num_key_value_heads
                );
                if config.is_quantized() {
                    println!(
                        "  Quantized:   {}-bit, group_size={}",
                        config.quantization_bits, config.quantization_group_size
                    );
                }
                println!();
            }
            Err(e) => {
                eprintln!("Warning: failed to parse config.json: {e}");
            }
        }
    }

    // Discover safetensors files
    match llama::weights::discover_weight_files(path) {
        Ok(files) => {
            println!("Safetensors files ({}):", files.len());
            for file in &files {
                let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
                println!(
                    "  {} ({:.1} GB)",
                    file.file_name().unwrap_or_default().to_string_lossy(),
                    size as f64 / 1e9,
                );
            }

            // Parse first file for tensor listing
            if let Some(first) = files.first() {
                match llama::weights::SafetensorsFile::open(first) {
                    Ok(sf) => {
                        let mut names: Vec<&str> =
                            sf.header.tensors.keys().map(|s| s.as_str()).collect();
                        names.sort();

                        println!();
                        println!("Tensors in first file ({}):", names.len());
                        for name in names.iter().take(20) {
                            if let Some(info) = sf.header.tensors.get(*name) {
                                println!(
                                    "  {}: {} {:?} ({:.1} MB)",
                                    name,
                                    info.dtype,
                                    info.shape,
                                    info.byte_size() as f64 / 1e6,
                                );
                            }
                        }
                        if names.len() > 20 {
                            println!("  ... and {} more", names.len() - 20);
                        }

                        // Show metadata
                        if !sf.header.metadata.is_empty() {
                            println!();
                            println!("Metadata:");
                            for (k, v) in &sf.header.metadata {
                                println!("  {k}: {v}");
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to parse {}: {e}", first.display());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

/// Start an interactive chat session.
fn cmd_chat(args: &[String]) {
    let model_dir = match args.first() {
        Some(dir) => dir,
        None => {
            eprintln!(
                "Usage: llama-cli chat <MODEL_DIR> [--temperature T] [--top-p P] [--top-k K]"
            );
            std::process::exit(1);
        }
    };

    // Parse sampling options
    let mut sampling = llama::sampling::SamplingConfig::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--temperature" | "-t" => {
                i += 1;
                if let Some(val) = args.get(i).and_then(|s| s.parse().ok()) {
                    sampling.temperature = val;
                }
            }
            "--top-p" => {
                i += 1;
                if let Some(val) = args.get(i).and_then(|s| s.parse().ok()) {
                    sampling.top_p = val;
                }
            }
            "--top-k" => {
                i += 1;
                if let Some(val) = args.get(i).and_then(|s| s.parse().ok()) {
                    sampling.top_k = val;
                }
            }
            _ => {}
        }
        i += 1;
    }

    println!("llama-rs v{} interactive chat", llama::VERSION);
    println!("Model: {model_dir}");
    println!(
        "Sampling: temperature={}, top_p={}, top_k={}",
        sampling.temperature, sampling.top_p, sampling.top_k
    );
    println!("Type /quit to exit, /reset to start a new conversation.");
    println!();

    // Attempt to create a session
    let tokenizer_path = std::path::Path::new(model_dir).join("tokenizer.json");
    let session_config = llama::session::SessionConfig {
        model_path: model_dir.to_string(),
        tokenizer_path: tokenizer_path.to_string_lossy().to_string(),
        max_seq_len: 8192,
        sampling,
    };

    match llama::Session::new(session_config) {
        Ok(mut session) => {
            session.set_system_prompt("You are a helpful, harmless, and honest assistant.");
            interactive_loop(&mut session);
        }
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            eprintln!();
            eprintln!("Note: This requires Apple Silicon with MLX installed.");
            eprintln!("Model directory should contain:");
            eprintln!("  - config.json");
            eprintln!("  - tokenizer.json");
            eprintln!("  - *.safetensors (model weights)");
            std::process::exit(1);
        }
    }
}

/// Run the interactive chat loop.
fn interactive_loop(session: &mut llama::Session) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("Read error: {e}");
                break;
            }
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/quit" | "/exit" => break,
            "/reset" => {
                if let Err(e) = session.reset() {
                    eprintln!("Reset error: {e}");
                }
                println!("[Conversation reset]");
                continue;
            }
            _ => {}
        }

        // Send message and stream response
        if let Err(e) = session.send_message(input) {
            eprintln!("Error: {e}");
            continue;
        }

        // Stream tokens
        loop {
            match session.wait_event() {
                Some(llama::engine::actor::ActorEvent::TokenBatch(tokens)) => {
                    let text = session.decode_tokens(&tokens);
                    print!("{text}");
                    stdout.flush().unwrap();
                }
                Some(llama::engine::actor::ActorEvent::Done {
                    tokens_per_second, ..
                }) => {
                    println!();
                    println!("[{tokens_per_second:.1} tokens/s]");
                    break;
                }
                Some(llama::engine::actor::ActorEvent::Error(e)) => {
                    eprintln!("\nError: {e}");
                    break;
                }
                Some(llama::engine::actor::ActorEvent::Stopped) => break,
                None => break,
            }
        }
    }
}

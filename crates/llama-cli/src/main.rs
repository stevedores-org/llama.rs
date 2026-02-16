use clap::{Parser, Subcommand};
use llama_engine::{LlamaEngine, Session};
use llama_runtime::MockEngine;

#[derive(Parser)]
#[command(name = "llama-cli")]
#[command(about = "CLI runner for llama.rs", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt (Milestone A Demo)
    Generate {
        /// The prompt to generate from
        #[arg(short, long)]
        prompt: String,
    },
}

use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Generate { prompt } => {
            let engine = MockEngine::new();
            let mut session = Session::new();

            println!("--- Milestone A Inference Demo ---");
            println!("Prompt: \"{}\"", prompt);

            // 1. Tokenize
            let tokens = engine.tokenize(prompt)?;

            // 2. Prefill
            engine.prefill(&mut session, &tokens)?;

            // 3. Decode (Streaming)
            // For the mock, we simulate multiple decode steps
            println!("Processing...");

            print!("Response: ");
            io::stdout().flush()?;

            for _ in 0..5 {
                let stream = engine.decode(&mut session)?;
                for token_result in stream {
                    let token = token_result?;
                    let text = engine.detokenize(&[token])?;
                    print!("{} ", text);
                    io::stdout().flush()?;
                }
            }
            println!("\n----------------------------------");
        }
    }

    Ok(())
}

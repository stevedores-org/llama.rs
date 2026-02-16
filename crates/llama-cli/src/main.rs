use clap::Parser;
use llama_cli::generate;

/// llama.rs — tiny model inference demo
#[derive(Parser)]
#[command(name = "llama-cli")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Generate text from a prompt using the tiny demo model.
    Generate {
        /// Input prompt text.
        #[arg(short, long)]
        prompt: String,

        /// Maximum number of tokens to generate.
        #[arg(short, long, default_value_t = 16)]
        max_tokens: usize,

        /// Random seed for reproducible sampling.
        #[arg(short, long, default_value_t = 42)]
        seed: u64,

        /// Sampling temperature (higher = more random, must be > 0).
        #[arg(short, long, default_value_t = 1.0)]
        temperature: f32,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Generate {
            prompt,
            max_tokens,
            seed,
            temperature,
        } => {
            if temperature <= 0.0 {
                eprintln!("error: temperature must be positive (got {temperature})");
                std::process::exit(1);
            }
            match generate(&prompt, max_tokens, seed, temperature) {
                Ok(result) => {
                    println!("{}", result.text);
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}

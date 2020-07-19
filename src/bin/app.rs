use chrono::Local;
use icfp2020::api;
use icfp2020::galaxy;
use icfp2020::{Context as _, Result};
use std::io::Write as _;
use structopt::StructOpt;

fn parse_i32_with_context(s: &str) -> Result<i32> {
    s.parse().with_context(|| format!("can not parse: {}", s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_test_dummy() {
        assert_eq!(0, 0);
        assert_eq!(parse_i32_with_context("1").unwrap(), 1);
    }
}

#[derive(StructOpt, Debug)]
struct Cli {
    /// Sets the level of verbosity
    #[structopt(short = "v", parse(from_occurrences))]
    verbose: u64,
    #[structopt(subcommand)]
    cmd: Command,
}

#[derive(StructOpt, Debug)]
enum Command {
    #[structopt(name = "hello")]
    Hello {
        #[structopt(name = "arg")]
        arg: String,
    },
    /// api test
    #[structopt(name = "api")]
    Api,
    /// Interact with galaxy
    #[structopt(name = "interact")]
    Interact,
}

fn env_logger_verbose_init() {
    env_logger::builder()
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {:5} {}] ({}:{}) {}",
                Local::now().format("%+"),
                // record.level(),
                buf.default_styled_level(record.level()),
                record.target(),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args(),
            )
        })
        .init();
}

fn main() -> Result<()> {
    println!("Hello, world!");

    let args = Cli::from_args();
    if args.verbose > 0 {
        env_logger_verbose_init();
    } else {
        env_logger::init();
    }

    log::error!("error");
    log::warn!("warn");
    log::info!("info");
    log::debug!("debug");
    log::trace!("trace");

    match args.cmd {
        Command::Hello { arg } => {
            println!("Hello {}", arg);
            assert_eq!(parse_i32_with_context("1")?, 1);
        }
        Command::Api => api::test()?,
        Command::Interact => galaxy::run()?,
    }
    Ok(())
}

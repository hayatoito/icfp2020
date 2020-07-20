use chrono::Local;
use icfp2020::api;
use icfp2020::galaxy;
use icfp2020::interact;
use icfp2020::Result;
use std::io::Write as _;
use structopt::StructOpt;

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
    /// api test
    #[structopt(name = "api")]
    Api,
    /// Interact with galaxy with browser UI
    #[structopt(name = "interact")]
    Interact { port: Option<u16> },
    #[structopt(name = "bench")]
    Bench,
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
        Command::Api => api::test()?,
        Command::Bench => galaxy::bench()?,
        Command::Interact { port } => {
            let mut rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(interact::interact(port.unwrap_or(9999)))?;
        }
    }
    Ok(())
}

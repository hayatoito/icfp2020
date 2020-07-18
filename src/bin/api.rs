use icfp2020::{Context as _, Result};
use reqwest::StatusCode;

use chrono::Local;
use std::io::Write as _;
use std::path::PathBuf;
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
    #[structopt(name = "test")]
    Test,
}

#[allow(dead_code)]
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

fn json_pretty_print(raw_json: &str) -> Result<()> {
    let obj: serde_json::Value = serde_json::from_str(raw_json).unwrap();
    let s = serde_json::to_string_pretty(&obj).unwrap();
    println!("pretty: {}", s);
    Ok(())
}

fn get(server_url: &str, apikey: &str, api: &str) -> Result<()> {
    let url = format!("{}{}?apikey={}", server_url, api, apikey);
    println!("get url: {}", url);
    let response = reqwest::blocking::get(&url)?;
    match response.status() {
        StatusCode::OK => {
            json_pretty_print(&response.text()?)?;
        }
        _ => {
            println!("Unexpected server response:");
            println!("HTTP code: {}", response.status());
            json_pretty_print(&response.text()?)?;
        }
    }
    Ok(())
}

fn post(server_url: &str, apikey: &str, api: &str, body: String) -> Result<()> {
    let url = format!("{}{}?apikey={}", server_url, api, apikey);
    println!("post url: {}, body: {}", url, body);

    let client = reqwest::blocking::Client::new();
    let response = client.post(&url).body(body).send()?;
    match response.status() {
        StatusCode::OK => {
            println!("{}", response.text()?);
        }
        _ => {
            println!("Unexpected server response:");
            println!("HTTP code: {}", response.status());
            println!("{}", response.text()?);
        }
    }
    Ok(())
}

fn apikey() -> Result<String> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("task/apikey");
    Ok(std::fs::read_to_string(path)?.trim().to_string())
}

fn test() -> Result<()> {
    let server_url = "https://icfpc2020-api.testkontur.ru";
    let apikey = apikey()?;

    get(server_url, &apikey, "/submissions")?;
    get(server_url, &apikey, "/teams/current")?;
    post(server_url, &apikey, "/aliens/send", "0".to_string())?;

    Ok(())
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
        Command::Test => {
            test()?;
        }
    }
    Ok(())
}

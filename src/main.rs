use reqwest::StatusCode;
use std::io::Read;

type Result<T> = anyhow::Result<T>;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    assert!(args.len() >= 3);

    let server_url = &args[1];
    let player_key = &args[2];

    println!("ServerUrl: {}; PlayerKey: {}", server_url, player_key);

    let client = reqwest::blocking::Client::new();
    let mut response = client.post(server_url).body(player_key.clone()).send()?;

    match response.status() {
        StatusCode::OK => {
            let mut content = String::new();
            response.read_to_string(&mut content)?;
            println!("{}", content);
        }
        _ => {
            println!("Unexpected server response:");
            println!("HTTP code: {}", response.status());
            let mut content = String::new();
            response.read_to_string(&mut content)?;
            println!("{}", content);
            std::process::exit(2);
        }
    }

    Ok(())
}

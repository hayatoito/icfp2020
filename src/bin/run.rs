use std::io::Read;

type Result<T> = anyhow::Result<T>;

fn main() -> Result<()> {
    println!("Hello, world!");
    let args: Vec<String> = std::env::args().collect();

    let server_url = &args[1];
    let player_key = &args[2];

    println!("ServerUrl: {}; PlayerKey: {}", server_url, player_key);

    let client = reqwest::blocking::Client::new();
    let mut response = client.post(server_url).body(player_key.clone()).send()?;
    assert!(response.status().is_success());

    let mut content = String::new();
    response.read_to_string(&mut content).unwrap();

    println!("{}", content);
    Ok(())
}

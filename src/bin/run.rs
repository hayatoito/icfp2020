type Result<T> = anyhow::Result<T>;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let server_url = &args[1];
    let player_key = &args[2];
    println!("ServerUrl: {}; PlayerKey: {}", server_url, player_key);
    let response = reqwest::blocking::get(&format!("{}?playerKey={}", server_url, player_key))?;
    assert!(response.status().is_success());
    Ok(())
}

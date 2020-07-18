use reqwest::StatusCode;

type Result<T> = anyhow::Result<T>;

// https://icfpc2020-api.testkontur.ru/swagger/index.html

fn first_post(server_url: &str, player_key: &str) -> Result<()> {
    let client = reqwest::blocking::Client::new();
    let response = client
        .post(server_url)
        .body(player_key.to_string())
        .send()?;
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    assert!(args.len() >= 3);

    let server_url = &args[1];
    let player_key = &args[2];

    println!("ServerUrl: {}; PlayerKey: {}", server_url, player_key);

    first_post(server_url, player_key)?;
    Ok(())
}

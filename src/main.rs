use reqwest::StatusCode;

type Result<T> = anyhow::Result<T>;

// https://icfpc2020-api.testkontur.ru/swagger/index.html

fn get(server_url: &str, player_key: &str, api: &str) -> Result<()> {
    let url = format!("{}{}?apikey={}", server_url, api, player_key);
    println!("get url: {}", url);
    let response = reqwest::blocking::get(&url)?;
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

fn post(server_url: &str, player_key: &str, api: &str, body: String) -> Result<()> {
    let url = format!("{}/{}?apikey={}", server_url, api, player_key);
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    assert!(args.len() >= 3);

    let server_url = &args[1];
    let player_key = &args[2];

    println!("ServerUrl: {}; PlayerKey: {}", server_url, player_key);

    get(server_url, player_key, "/submissions")?;
    get(server_url, player_key, "/teams/current")?;
    post(server_url, player_key, "/alians/send", "0".to_string())?;

    Ok(())
}

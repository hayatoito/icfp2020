use crate::prelude::*;
use reqwest::StatusCode;

fn json_pretty_print(raw_json: &str) -> Result<()> {
    let obj: serde_json::Value = serde_json::from_str(raw_json).unwrap();
    let s = serde_json::to_string_pretty(&obj).unwrap();
    println!("pretty: {}", s);
    Ok(())
}

fn print_get(server_url: &str, apikey: &str, api: &str) -> Result<()> {
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

fn print_post(server_url: &str, apikey: &str, api: &str, body: String) -> Result<()> {
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

pub fn test() -> Result<()> {
    let server_url = "https://icfpc2020-api.testkontur.ru";
    let apikey = apikey()?;

    print_get(server_url, &apikey, "/submissions")?;
    print_get(server_url, &apikey, "/teams/current")?;
    print_post(server_url, &apikey, "/aliens/send", "0".to_string())?;

    Ok(())
}

fn post(server_url: &str, apikey: &str, api: &str, body: String) -> Result<String> {
    let url = format!("{}{}?apikey={}", server_url, api, apikey);
    info!("post url: {}, body: {}", url, body);

    let client = reqwest::blocking::Client::new();
    let response = client.post(&url).body(body).send()?;
    match response.status() {
        StatusCode::OK => Ok(response.text()?),
        _ => {
            error!("Unexpected server response:");
            error!("HTTP code: {}", response.status());
            bail!("response error: {} {}", response.status(), response.text()?)
        }
    }
}

pub fn send(s: String) -> Result<String> {
    let server_url = "https://icfpc2020-api.testkontur.ru";
    let apikey = apikey()?;
    post(server_url, &apikey, "/aliens/send", s)
}

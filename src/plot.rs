use crate::prelude::*;

use anyhow::Result;
use chrono::prelude::*;
use serde::Serialize;
use serde_json::json;
use std::io::Write;
use std::path::Path;

// Use plotly.js
// https://plot.ly/javascript/

// cheat sheet
// https://images.plot.ly/plotly-documentation/images/plotly_js_cheat_sheet.pdf

// TODO: Support multiple plots div elements.
// Use macro? e.g. write_html!(path, data1, data2, ....)
fn write_html<T: ?Sized>(path: impl AsRef<Path>, data: &T) -> Result<()>
where
    T: Serialize,
{
    let html = format!(
        r###"
<!DOCTYPE html>
<head>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<!-- <div id="myDiv" style="width:1280px;height:720px"></div> -->
<div id="myDiv" style="width:800px;height:600px"></div>
<script>
const myDiv = document.getElementById('myDiv');
Plotly.newPlot(myDiv, {});
</script>
    "###,
        serde_json::to_string(&data).unwrap(),
    );
    let mut file = std::fs::File::create(path.as_ref())?;
    file.write_all(html.as_bytes())?;
    Ok(())
}

fn unique_file_name() -> String {
    let local: DateTime<Local> = Local::now();
    local.format("%Y-%m-%d-%H%M%S-%f").to_string()
}

pub fn plot<T>(data: &T) -> Result<()>
where
    T: Serialize,
{
    let mut path = std::env::temp_dir();
    path.push(&format!("plot-{}.html", unique_file_name()));
    println!("plot: {}", path.display());
    write_html(&path, data)?;
    webbrowser::open(path.to_str().unwrap())?;
    // Ok(path)
    Ok(())
}

pub fn plot_galaxy(points: Vec<(i64, i64)>) -> Result<()> {
    let trace = json!({
        "x": points.iter().map(|p| p.0).collect::<Vec<_>>(),
        "y": points.iter().map(|p| p.1).collect::<Vec<_>>(),
        "mode": "markers",
    });
    plot(&[trace])
}

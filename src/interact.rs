use crate::galaxy;
use crate::prelude::*;
pub use log::*;
use std::convert::Infallible;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use warp::Filter;

type Db = Arc<Mutex<u64>>;

pub async fn interact(port: u16) -> Result<()> {
    let addr = ([127, 0, 0, 1], port);

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("task/galaxy.txt");
    let src = std::fs::read_to_string(path)?.trim().to_string();

    let (click_sender, click_receiver) = mpsc::channel();
    let (screen_sender, screen_receiver) = mpsc::channel();

    tokio::task::spawn_blocking(move || {
        let mut galaxy = galaxy::Galaxy::new(&src).unwrap();
        galaxy.play(click_receiver, screen_sender).unwrap();
    });

    let screen_data = Arc::new(Mutex::new(Screen::new()));

    let screen1 = screen_data.clone();
    tokio::task::spawn_blocking(move || update_screen(screen1, screen_receiver));

    let db: Db = Arc::new(Mutex::new(0));

    let click_sender = Arc::new(Mutex::new(click_sender));

    let mut static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    static_dir.push("task");

    let route = count(db)
        .or(click(click_sender))
        .or(screen(screen_data))
        .or(warp::fs::dir(static_dir))
        .with(warp::log::custom(|info| {
            debug!("{} {} {}", info.method(), info.path(), info.status());
        }));

    println!("Open http://localhost:{}/galaxypad.html to play.", port);
    warp::serve(route).run(addr).await;
    Ok(())
}

// Runs on another thead
fn update_screen(screen: Arc<Mutex<Screen>>, screen_receriver: ScreenReceiver) {
    while let Ok(data) = screen_receriver.recv() {
        let mut screen = screen.lock().unwrap();
        *screen = data;
    }
}

pub fn count(db: Db) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path!("count")
        .map(move || db.clone())
        .and_then(inc_count)
}

async fn inc_count(db: Db) -> Result<impl warp::Reply, Infallible> {
    let mut db = db.lock().unwrap();
    *db += 1;
    info!("db: {:?}", *db);
    Ok(format!("ok: db is {}", *db))
}

pub fn click(
    click_sender: Arc<Mutex<ClickSender>>,
) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path!("click" / i64 / i64)
        .map(move |x, y| (x, y, click_sender.clone()))
        .and_then(send_click)
}

async fn send_click(
    (x, y, click_sender): (i64, i64, Arc<Mutex<ClickSender>>),
) -> Result<impl warp::Reply, Infallible> {
    info!("click: {:?}", (x, y));
    let sender = click_sender.lock().unwrap();
    sender.send((x, y)).unwrap();
    Ok(format!("ok: click is {:?}", (x, y)))
}

pub fn screen(
    screen_data: Arc<Mutex<Screen>>,
) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path!("screen")
        .map(move || screen_data.clone())
        .and_then(serve_screen)
}

async fn serve_screen(screen_data: Arc<Mutex<Screen>>) -> Result<impl warp::Reply, Infallible> {
    let screen_data = screen_data.lock().unwrap();
    Ok(warp::reply::json(&*screen_data))
}

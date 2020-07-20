// pub use lazy_static::*;
pub use anyhow::{bail, ensure, Context, Result};
pub use log::*;
pub use std::collections::{HashMap, HashSet, VecDeque};
pub use std::path::{Path, PathBuf};
pub use std::rc::Rc;

pub type Click = (i64, i64);
pub type Screen = Vec<(i64, i64)>;

pub type ScreenSender = std::sync::mpsc::Sender<Screen>;
pub type ScreenReceiver = std::sync::mpsc::Receiver<Screen>;

pub type ClickSender = std::sync::mpsc::Sender<Click>;
pub type ClickReceiver = std::sync::mpsc::Receiver<Click>;

#[cfg(test)]
mod tests {
    #[test]
    fn prelude_test_dummy() {
        assert_eq!(0, 0);
    }
}
